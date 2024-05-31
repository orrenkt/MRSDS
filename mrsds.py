"""
Multi-Region Switching Dynamical Systems (MR-SDS)
Mean field inference.
"""
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import time

from mrsds.generative_models import (ContinuousStateTransition, DiscreteStateTransition,
                                     GaussianEmissions, PoissonEmissions,
                                     build_mr_dynamics_net, build_mr_emission_net,
                                     construct_x0_dist)
import mrsds.forward_backward as fb
from mrsds.inference import TransformerInferenceNetwork, GaussianPosterior, build_mr_embedding_net
from mrsds.utils_tensor import normalize_logprob, mask_logprob
from mrsds.utils_config import attr_dict_recursive, get_distribution_config
from mrsds.networks import build_birnn, build_dense_net, build_transformer


class MRSDS(tf.keras.Model):
  """
  """
  def __init__(self, x_transition_net, z_transition_net, emission_nets,
               inference_net, x0_distribution=None,
               continuous_state_dim=None, num_inputs=2, num_states=None,
               sz=False, uz=False, xz=True, ux=True, dtype=tf.float32):
    """
    TBD
    """
    super(MRSDS, self).__init__()

    self.x_tran = x_transition_net
    self.z_tran = z_transition_net
    self.y_emit = emission_nets
    self.inference_net = inference_net

    self.num_inputs = num_inputs

    self.num_states = num_states
    if num_states is None:
      self.num_states = self.z_tran.output_event_dims

    self.x_dim = continuous_state_dim
    if continuous_state_dim is None:
      self.x_dim = self.x_tran.output_event_dims

    self.x0_dist = x0_distribution
    ones = tf.ones([self.num_states], dtype=dtype)
    self.log_init_z = tf.Variable(normalize_logprob(ones, axis=-1)[0])

    # Controls if inputs u, stages s, latents x -> ztran; u -> xtran
    self.uz = uz
    self.sz = sz
    self.xz = xz
    self.ux = ux

  def call(self, ys, us=None, masks=None, drop_idxs=None, num_samples=1,
           dtype=tf.float32, do_batch=False, random_seed=0, xmean=False,
           temperature=1.0, beta=1):
    """
    Inference call of MRSDS.

    Args:
      inputs: float Tensor [batch_size, num_steps, event_size]
        Data observations Y
      temperature: float scalar. Default: 1.0
        Controls discrete transition z_t ~ Softmax( p(z[t]|z[t-1], x[t-1]) / temp)
        This gets annealed over the course of training. Increasing temperature
        increases the posterior entropy (ie certainty about) discrete states.
        Used to encourage the model to occupy all the states early on in training.
      num_samples: int scalar
        Number of posterior samples drawn from inference net x[i] ~ q(x[1:T] | y[1:T])
      dtype: data type. Default: tf.float32

    Returns:
      return_dict: python `dict`
        Result dict with result tensors as values. Keys:
          x_sampled: posterior samples x^[1:T] ~ q(x|y)
          x_entropy: entropy of posterior samples H[q(x^[t] | y[1:T])] = E[-log q(x^)]
          z_posterior: prob log_a(j,k) = p(z[t]=j | z[t-1]=k, x[t-1])
          z_posterior_ll: log p(z[t] | z[1:T], x[1:T])
          log_P: explicit duration count probs p(c[t] | z[t], c[t]). Optional
          elbo: Evidence lower bound
          sequence_likelihood: complete data log likelihood p(z^[1:T], x^[1:T], y[0:T])
          reconstructed_ys: reconstructed inputs y^ ~ p(y|z^,x^)
          ys: data inputs y, provided for convenience
          us: data inputs u, provided for convenience
    """
    tf.random.set_seed(random_seed)

    inputs, masks, time_masks = self._prep_inputs(ys, us, masks, dtype)
    num_samples = tf.convert_to_tensor(num_samples, dtype_hint=tf.int32)
    batch_size, num_steps = tf.unstack(tf.shape(inputs[0])[:2])

    time_masks_sample = tf.repeat(time_masks,[num_samples], axis=0)

    # Mask inputs for inference net.
    inputs_masked = [inputs[0] * masks, inputs[1]]

    inputs = [tf.repeat(_,[num_samples],axis=0) for _ in inputs]

    # Sample continuous latent x ~ q(x[1:T] | y[1:T])
    x_sampled, x_entropy, _ = self.inference_net(inputs_masked,
                                                 num_samples=num_samples,
                                                 masks=time_masks,
                                                 random_seed=random_seed)

    # Merge batch_size and num_samples dimensions (typically samples=1).
    shapes = [num_samples*batch_size, num_steps]
    x_dim = tf.shape(x_sampled)[-1]
    x_sampled = tf.reshape(x_sampled, [*shapes, x_dim])
    #x_entropy = tf.reshape(x_entropy, shapes)

    # Mask the samples
    if masks is not None:
      x_sampled *= time_masks_sample[:,:,tf.newaxis]
      x_entropy *= time_masks[:,:]

    # Reshape for samples
    #inputs_ = [tf.tile(_, [num_samples,1,1]) for _ in inputs]

    # Initial discrete state
    log_init_z = self.log_init_z

    # Based on sampled continuous states x_sampled, get discrete and continuous
    # state likelihoods.
    # log_a(j, k) = p(z[t]=j | z[t-1]=k, x[t-1], u[t])
    # log_b(k) = p(x[t] | x[t-1], u[t], z[t]=k)
    log_a = self.compute_z_transition_probs(x_sampled, inputs[1])
    log_b, xt1_gen_dist = self.compute_x_likelihood(x_sampled, inputs[1], time_masks_sample)
    if masks is not None:
      log_b = mask_logprob(log_b, time_masks_sample[:,:,tf.newaxis], 'b')
      log_a = mask_logprob(log_a, time_masks_sample[:,:,tf.newaxis,tf.newaxis], 'a')

    # Forward-backward algorithm returns posterior marginal of discrete states
    # `log_gamma1 = p(z[t]=k | x[1:T], u[1:T])'
    # `log_gamma2 = p(z[t]=k, z[t-1]=j | x[1:T], u[1:T])'
    fwd, bwd, log_gamma1, log_gamma2, log_px = fb.forward_backward(log_a, log_b, log_init_z,
                                                                   time_masks_sample,
                                                                   temp=temperature)
    logpx_sum = tf.reduce_mean(log_px, axis=0)

    # Marginalize over z to get t1 gen xs
    xt1_gen_mean = xt1_gen_dist.mean() * tf.math.exp(log_gamma1[:,1:,:,tf.newaxis])
    #xt1_gen_mean = tf.squeeze(tf.reduce_sum(xt1_gen_mean, axis=-2)) # breaks for k=1
    xt1_gen_mean = tf.reduce_sum(xt1_gen_mean, axis=-2)
    x_sampled_gen1 = tf.concat([x_sampled[:,:1,:], xt1_gen_mean], axis=1)

    # Compute 1-step gen emissions and likelihoods p(y[1:T] | xinf[1], x1gen[2:T])
    res = self.compute_y_likelihood(inputs, x_sampled_gen1, time_masks_sample, masks)
    y_dist, log_py, log_py_cosmooth = res
    ys_recon = y_dist.mean()
    if masks is not None:
      ys_recon *= time_masks_sample[:,:,tf.newaxis]

    # Compute ELBO, complete data likelihood and entropy
    log_p_xy = log_px + tf.reduce_sum(log_py, axis=-1)
    elbo, log_pxy1T, log_qx = self.compute_objective(x_entropy, log_p_xy)

    print('logpx_sum', logpx_sum.shape, logpx_sum)
    print('logpxy', log_p_xy.shape, tf.reduce_sum(log_p_xy))
    print('elbo', elbo.shape, elbo)

    # Count the number of state switches - just to track
    zs = tf.argmax(log_gamma1, axis=-1)
    switches = tf.cast(tf.math.not_equal(zs[:,:-1],zs[:,1:]), tf.float32)
    num_zswitches = tf.reduce_sum(switches)

    # Reshape for samples if any
    _ = [batch_size, num_steps, -1]
    ys_recon = tf.reshape(ys_recon, _)
    log_gamma1 = tf.reshape(log_gamma1, _)
    x_dim = tf.shape(x_sampled)[-1]
    x_sampled = tf.reshape(x_sampled, [*_, x_dim])
    log_py_sum = tf.reduce_mean(tf.reduce_sum(log_py, axis=1), axis=0)

    # Compute smoothness
    diffs = x_sampled[:,:1,:] - x_sampled[:,1:,:]
    mean_norm = tf.reduce_mean(tf.norm(x_sampled,axis=-1))
    diffs_norm = tf.reduce_mean(tf.norm(diffs,axis=-1)) / mean_norm

    return_dict = dict(
     [("ys", inputs[0]),
      ("us", inputs[1]),
      ("reconstructed_ys", ys_recon),
      ("log_py", log_py),
      ("logpy_sum", log_py_sum),
      ("log_px", log_px),
      ("logpx_sum", logpx_sum),
      ("log_py_cosmooth", log_py_cosmooth),
      ("z_posterior", log_a),
      ("z_posterior_ll", log_gamma1),
      ("num_zswitches", num_zswitches),
      ("x_sampled", x_sampled),
      ("elbo", elbo),
      ("diffs_norm", diffs_norm),
      ("sequence_likelihood", log_pxy1T),
      ("xt_entropy", log_qx),
      ("log_a", log_a),
      ("log_b", log_b)])

    return return_dict

  def _prep_inputs(self, inputs, us, masks, dtype=tf.float32):
    """
    Read in masks and inputs.
    Mask is for specifying any padded / dropped neurons.
    Time mask is for masking timepoints only.
    """
    inputs = tf.convert_to_tensor(inputs, dtype_hint=dtype, name="MRSDS_Input_Tensor")
    batch_size, num_steps = tf.unstack(tf.shape(inputs)[:2])
    if masks is not None:
      masks = tf.convert_to_tensor(masks, dtype_hint=dtype, name="MRSDS_Mask_Tensor")
      # Add a time dim if not given
      if len(tf.unstack(tf.shape(masks))) == 2:
        masks = tf.expand_dims(masks, axis=1)
        masks = tf.repeat(masks, repeats=[num_steps], axis=1)
    else:
      masks = tf.ones_like(inputs, dtype=dtype)
      print('masks none. null mask:', inputs.shape, masks.shape)
    time_masks = tf.squeeze(masks[:,:,0])

    if us is not None:
      us = tf.convert_to_tensor(us, dtype_hint=dtype, name="MRSDS_US_Tensor")
    else:
      us = tf.zeros([batch_size, num_steps, self.num_inputs], dtype=dtype)

    inputs = [inputs, us]
    return inputs, masks, time_masks

  def compute_objective(self, log_qx, log_p_xy):
    """
    Compute ELBO
    """
    log_qx = tf.reduce_mean(tf.reduce_sum(log_qx, axis=1), axis=0)
    log_p_xy = tf.reduce_mean(log_p_xy, axis=0)
    elbo = log_p_xy + log_qx
    return elbo, log_p_xy, log_qx

  def compute_z_transition_probs(self, x_sampled, us, temperature=1.0):
    """
    Compute log p(z[t] | z[t-1], x[t-1]).

    Args:
      x_sampled: float tensor [batch_size, num_steps, latent_dim]
        Continuous latent x ~ q(x|y)
      temperature: float scalar
        Temperature for transition probability p(z[t] | z[t-1], x[t-1]).

    Returns:
      prob_zt_ztm1: float tensor [batch_size, num_steps, num_states, num_states]
        Transition probability p(z_t | z_t-1, x_t-1).
    """
    batch_size, num_steps = tf.unstack(tf.shape(x_sampled)[:2])

    # Get z transition probs
    ztransition_inputs = None
    if self.xz:
      ztransition_inputs = x_sampled[:,:-1,:]
    else:
      print('xs not passed to ztransition.')
    if self.uz: # NOTE Input driven switches case
      if ztransition_inputs is None:
        ztransition_inputs = us[:,1:,:]
      else:
        ztransition_inputs = tf.concat([ztransition_inputs, us[:,1:,:]], axis=-1)
      print('us passed to ztransition.')

    # Get unnormalized discrete state transition matrix
    z_sample = self.z_tran(ztransition_inputs)
    if num_steps is not None:
      sizes = [batch_size, num_steps-1, self.num_states, self.num_states]
      log_prob_zt_ztm1 = tf.reshape(z_sample, sizes)
    else:
      log_prob_zt_ztm1 = z_sample

    # Normalizing -2 axis makes transition from state j p(:,j) a proper dist
    log_prob_zt_ztm1 = normalize_logprob(log_prob_zt_ztm1, axis=-2,
                                         temperature=temperature)[0]
    eps = 1e-10  # to prevent inf in log
    eye = tf.eye(self.num_states, self.num_states, batch_shape=[batch_size,1]) + eps
    log_prob_zt_ztm1 = tf.concat([tf.math.log(eye), log_prob_zt_ztm1], axis=1)

    return log_prob_zt_ztm1

  def compute_x_likelihood(self, x_sampled, us, time_masks):
    """
    """
    # --- Compute log p(x[t] | x[t-1], z[t], u[t]) ---

    x0_dist = self.x0_dist

    # log_prob_x0 is [batch_size, num_states]
    x_sampled0 = x_sampled[:,0,:]
    log_prob_x0 = x0_dist.log_prob(x_sampled0[:,tf.newaxis,:])[:, tf.newaxis,:]

    x0mean = x0_dist.mean()

    # log_prob_xt is of shape [batch_size, num_steps, num_states]
    log_prob_xt, msgs, xt1_gen_dist = self.compute_x_transition_probs(x_sampled, log_prob_x0, us)

    # Mask out padded timepoints
    if time_masks is not None:
      if len(tf.unstack(tf.shape(log_prob_xt))) == 2:
        mask_ = time_masks
      else:
        mask_ = time_masks[:,:,tf.newaxis]
      log_prob_xt = mask_logprob(log_prob_xt, mask_, 'x')

    return log_prob_xt, xt1_gen_dist

  def compute_y_likelihood(self, inputs, x_sampled_gen1, time_masks=None, mask=None):
    """
    """
    if len(inputs) == 2:
      ys, us = inputs
    else:
      ys = inputs[0]
      us = None
    batch_size, num_steps, num_neurons = tf.unstack(tf.shape(ys)[:3])

    # --- Compute log p(y[t] | x[t]) ---

    emission_dist = self.y_emit(x_sampled_gen1)

    # `emission_dist' is same shape as `ys', but could be smaller if padded neurons
    # `log_prob_yt' is [batch_size, num_steps]
    ys = tf.reshape(ys[:,:,:num_neurons], [batch_size,num_steps,-1])
    log_prob_yt = emission_dist.log_prob(tf.squeeze(ys))

    if mask is not None and type(emission_dist) == PoissonEmissions:
      neurons_mask = tf.logical_not(tf.cast(mask, tf.bool))
      neurons_mask = tf.reshape(neurons_mask, [batch_size,num_steps,-1])
      cosmooth_dist = tfd.Masked(emission_dist, neurons_mask)
      ys_masked = ys * neurons_mask
      log_py_cosmooth = cosmooth_dist.log_prob(tf.squeeze(ys_masked))
    else:
      log_py_cosmooth = tf.identity(log_prob_yt)

    # Mask out padded timepoints
    if time_masks is not None:
      if len(tf.unstack(tf.shape(log_prob_yt))) == 2:
        mask_ = time_masks
      else:
        mask_ = time_masks[:,:,tf.newaxis]
      log_prob_yt = mask_logprob(log_prob_yt, mask_, 'y')

    # Catch for Poisson vs Gaussian emission shapes, double check event shape def
    log_py = tf.identity(log_prob_yt)
    if len(tf.unstack(tf.shape(log_prob_yt))) > 2:
      log_prob_yt = tf.reduce_sum(log_prob_yt, axis=-1)

    return emission_dist, log_py, log_py_cosmooth

  def compute_x_transition_probs(self, x_sampled, log_prob_x0, us=None):
    """
    p(x[t] | x[t-1], z[t]) transition.
    """
    if us is not None and self.ux:
      inputs_ = [x_sampled[:,:-1,:], us[:,1:,:]]
    else:
      nt, T = tf.unstack(tf.shape(x_sampled)[:2])
      inputs_ = [x_sampled[:,:-1,:], tf.zeros([nt,T-1,2])]

    prior_distributions = self.x_tran(*inputs_)
    future_tensor = x_sampled[:,1:,:]
    log_prob_xt = prior_distributions.log_prob(future_tensor[:,:,tf.newaxis,:])
    log_prob_xt = tf.concat([log_prob_x0, log_prob_xt], axis=1)
    messages = None
    return log_prob_xt, messages, prior_distributions

  def compute_x_transition_probs_multistep(self, x_sampled, log_prob_x0, us=None, T=3):
    """
    p(x[t] | x[t-1], z[t]) transition.
    """

    if not self.ux and us is None:
      print('Using zeros for u')
      nt, T = tf.unstack(tf.shape(x_sampled)[:2])
      us = tf.zeros([nt,T,2])

    inputs_ = [x_sampled[:,:-1,:], us[:,1:,:]]
    xpred_dist1 = self.x_tran(*inputs_)
    future_tensor = x_sampled[:,1:,:]
    log_prob_xt1 = xpred_dist1.log_prob(future_tensor[:,:,tf.newaxis,:])
    log_prob_xt1 = tf.concat([log_prob_x0, log_prob_xt1], axis=1)
    xmean1 = xpred_dist1.mean()

    inputs_ = [xmean1[:,:-1,:], us[:,2:-1,:]]
    xpred_dist2 = self.x_tran(*inputs_)
    future_tensor = x_sampled[:,2:,:]
    log_prob_xt2 = xpred_dist2.log_prob(future_tensor[:,:,tf.newaxis,:])
    log_prob_xt2 = tf.concat([log_prob_x0, log_prob_xt1[:,1:2,:], log_prob_xt2], axis=1)
    xmean2 = xpred_dist2.mean()

    inputs_ = [xmean2[:,:-1,:], us[:,3:-2,:]]
    xpred_dist3 = self.x_tran(*inputs_)
    future_tensor = x_sampled[:,3:,:]
    log_prob_xt3 = xpred_dist3.log_prob(future_tensor[:,:,tf.newaxis,:])
    log_prob_xt3 = tf.concat([log_prob_x0, log_prob_xt1[:,1:2,:],
                              log_prob_xt2[:,1:2,:], log_prob_xt3], axis=1)
    xmean3 = xpred_dist3.mean()

    xpred_dists = [xpred1, xpred2, xpred3]

    # Average the log probs
    weights = [0.333, 0.333, 0.333]
    log_prob_xt = log_prob_xt1*weights[0] + log_prob_xt2*weights[1] + log_prob_xt3*weights[2]

    messages = None
    return log_prob_xt, messages, xpred_dists


def build_model(model_dir, config_path, num_regions, num_dims,
                region_sizes, trial_length, num_states,
                name='mrsds', num_days=1, hist_dims=1,
                zcond=False, random_seed=0):

  print('model seed', random_seed)
  np.random.seed(random_seed)
  tf.random.set_seed(random_seed)

  with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
    config = attr_dict_recursive(config)
    cfd = config.data
    cfm = config.model
    cfi = config.inference
    cft = config.training

  cfmxd = cfm.xtransition.dynamics
  latent_region_sizes = [num_dims] * num_regions
  #num_regions = len(latent_region_sizes)
  hidden_dim = np.sum(latent_region_sizes)
  num_neurons = np.sum(region_sizes)
  if hasattr(cfmxd, "single_region_latent"):
    if cfmxd.single_region_latent:
      latent_region_sizes = [num_dims*num_regions]

  listzip = lambda x: list(zip(*x))

  ## ----- Generative model -----

  # --- Discrete transitions ---

  z_init_net = None

  # Net returns the transition prob
  # `log p(z[t] |z[t-1], x[t-1])`
  num_states_sq = num_states ** 2
  layer_mult = 2
  #if hasattr(cfm, 'ztransition'):
  #  layer_mult = cfm.ztransition.layer_mult
  #print('ztran layer_mult', layer_mult)
  dense_layers = [[layer_mult * num_states_sq, num_states_sq], ["relu", None]]
  #dense_layers = [[num_states_sq], [None]]
  net_z_transition = build_dense_net(listzip(dense_layers))
  z_transition = DiscreteStateTransition(
    transition_net=net_z_transition, num_states=num_states)

  # --- Continuous transitions ---

  xic_inf_net = None
  x0_dist = construct_x0_dist(
      latent_dims=hidden_dim,
      num_states=num_states,
      use_trainable_cov=cfm.xic.trainable_cov,
      use_triangular_cov=False, #cfm.xic.triangular_cov,
      name=name+"_xic")

  # Configuring p(x[t] | x[t-1], z[t])
  cxt = cfm.xtransition
  seq_len = np.max(trial_length)-1
  dynamics_layers = [[*cxt.dynamics.layers,
                      hidden_dim/len(latent_region_sizes)],
                      cxt.dynamics.activations]
  num_inputs = cfd.inputs
  input_layers = [[*cxt.inputs.layers, latent_region_sizes[0]],
                  cxt.inputs.activations]
  inputs_ = [latent_region_sizes, seq_len, num_inputs,
             listzip(dynamics_layers), listzip(input_layers)]
  cond_dict = {
    'communication_type': cxt.dynamics.communication_type,
    'input_type': cxt.dynamics.input_type,
    'input_func_type': cxt.dynamics.input_func_type,
  }
  xdropout = False
  if hasattr(cfm, 'xdropout'):
    cond_dict['xdropout'] = cfm.xdropout
    xdropout = cfm.xdropout
    print('using xdropout in gen model')
  cfg_xtr = get_distribution_config(triangular_cov=cxt.triangular_cov,
                                    trainable_cov=cxt.trainable_cov)

  layer_reg = False
  if hasattr(cxt, 'layer_reg'):
    layer_reg = cxt.layer_reg
  cond_dict['layer_reg'] = layer_reg

  if zcond:
    print('using zcond dynamics')
    from mrsds.generative_models import ContinuousStateTransitionZCond
    x_transition_nets = [build_mr_dynamics_net(*inputs_, **cond_dict, zcond=True,
                                               num_states=num_states)]
    x_transition = ContinuousStateTransitionZCond(
      transition_mean_net=x_transition_nets[0],
      distribution_dim=hidden_dim,
      num_states=num_states,
      use_triangular_cov=cfg_xtr.use_triangular_cov,
      use_trainable_cov=cfg_xtr.use_trainable_cov,
      raw_sigma_bias=cfg_xtr.raw_sigma_bias,
      sigma_min=cfg_xtr.sigma_min,
      sigma_scale=cfg_xtr.sigma_scale,
      ux=cfm.ux, name=name+"_x_trans")
  else:
    x_transition_nets = [build_mr_dynamics_net(*inputs_, **cond_dict, seed=random_seed+i)
                         for i in range(num_states)]
    x_transition = ContinuousStateTransition(
      transition_mean_nets=x_transition_nets,
      distribution_dim=hidden_dim,
      num_states=num_states,
      use_triangular_cov=cfg_xtr.use_triangular_cov,
      use_trainable_cov=cfg_xtr.use_trainable_cov,
      raw_sigma_bias=cfg_xtr.raw_sigma_bias,
      sigma_min=cfg_xtr.sigma_min,
      sigma_scale=cfg_xtr.sigma_scale,
      ux=cfm.ux, name=name+"_x_trans")

  # --- Emissions ---

  # Configuring emission distribution p(y[t] | x[t])
  ce = cfm.emissions
  config_emission = get_distribution_config(trainable_cov=ce.trainable_cov,
                                            triangular_cov=ce.triangular_cov)

  # Note the final emission layer size depends on the region
  dense_layers = [ce.layers, ce.activations]
  nl = None
  if ce.final_activation == 'exp':
    nl = tf.exp
  elif ce.final_activation == 'softplus':
    nl = tf.math.softplus
  final_layers = [region_sizes, [nl]*num_regions]

  if hasattr(cfmxd, "single_region_latent"):
    if cfmxd.single_region_latent:  # Still have multiregion emissions
      latent_region_sizes = [num_dims] * num_regions
      print('sizes', region_sizes, latent_region_sizes)

  layer_reg = False
  if hasattr(ce, 'layer_reg'):
    layer_reg = ce.layer_reg

  inputs = [latent_region_sizes, trial_length,
            listzip(dense_layers), listzip(final_layers),
            region_sizes, np.sum(region_sizes)]
  emission_net = build_mr_emission_net(*inputs, xdropout=xdropout, layer_reg=layer_reg)

  cfg_em = config_emission
  if ce.dist == "gaussian":
    emission_nets = GaussianEmissions(
      emission_mean_nets=emission_net,
      observation_dims=num_neurons,
      use_triangular_cov=cfg_em.use_triangular_cov,
      use_trainable_cov=cfg_em.use_trainable_cov,
      raw_sigma_bias=cfg_em.raw_sigma_bias,
      sigma_min=cfg_em.sigma_min,
      sigma_scale=cfg_em.sigma_scale,
      name=name+"_y_emit")
  elif ce.dist == "poisson":
    emission_nets = PoissonEmissions(
      emission_rate_nets=emission_net,
      observation_dims=num_neurons,
      name=name+"_y_emit")

  # ----- Inference -----

  cfg_inf = get_distribution_config(triangular_cov=cfi.triangular_cov,
                                    trainable_cov=cfi.trainable_cov)
  input_len = cfd.inputs
  emb = cfi.embedding
  dense_layers = [emb.layers, emb.activations]
  inputs = [region_sizes, trial_length,
            num_neurons, listzip(dense_layers)]
  embedding_net = build_mr_embedding_net(*inputs, input_len=input_len)

  posterior_distribution = GaussianPosterior(
    nets=[tf.identity],
    latent_dims=[hidden_dim],
    use_triangular_cov=cfg_inf.use_triangular_cov,
    use_trainable_cov=cfg_inf.use_trainable_cov,
    raw_sigma_bias=cfg_inf.raw_sigma_bias,
    sigma_min=cfg_inf.sigma_min,
    sigma_scale=cfg_inf.sigma_scale,
    name=name+"_posterior")

  cfit = cfi.transformer_params
  input_shape = (seq_len+1, emb.layers[-1]*num_regions + input_len,)
  transformer = build_transformer(input_shape, hidden_dim, cfit.head_size, cfit.num_heads,
                                  cfit.ff_dim, cfit.num_transformer_blocks, cfit.mlp_units)
  print(transformer.summary())
  posterior_net = TransformerInferenceNetwork(
    transformer=transformer,
    posterior_dist=posterior_distribution,
    latent_dims=hidden_dim,
    embedding_nets=embedding_net)

  # ----- Build MRSDS model -----

  mrsds_model = MRSDS(
    x_transition_net=x_transition,
    z_transition_net=z_transition,
    emission_nets=emission_nets,
    inference_net=posterior_net,
    x0_distribution=x0_dist,
    continuous_state_dim=None,
    num_states=None,
    num_inputs=num_inputs,
    uz=cfm.uz, xz=cfm.xz, sz=cfm.sz,
    ux=cfm.ux)

  shapes = (cft.batch_size, trial_length, num_neurons)
  mrsds_model.build(input_shape=shapes)
  print(mrsds_model.summary())

  xic_nets = None
  return mrsds_model, x_transition_nets, xic_nets


def load_model(model_dir, config_path, num_regions, num_dims,
               region_sizes, trial_length, num_states, num_inputs=0, load=True):

  from mrsds.mrsds import build_model
  from mrsds.utils_config import load_yaml_config

  #  NOTE This really shouldn't be duplicated with run_training, should make it's own func
  config = load_yaml_config(config_path)
  cfi = config.inference
  cfm = config.model
  cxt = cfm.xtransition
  cfd = config.data
  zcond = False
  if hasattr(cxt, 'zcond') and cxt.zcond:
    zcond = True

  # Build model and load checkpoint
  args = [num_regions, num_dims, region_sizes, trial_length]
  (mrsds_model, x_transition_networks,
   continuous_state_init_dist) = build_model(model_dir, config_path, *args,
                                             hist_dims=hist_dims,
                                             num_states=num_states, zcond=zcond)
  print('built')
  if not load:
    return mrsds_model, x_transition_networks, continuous_state_init_dist
  optimizer = tf.keras.optimizers.Adam()
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=mrsds_model)
  ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=5)
  latest_checkpoint = tf.train.latest_checkpoint(model_dir)
  if latest_checkpoint:
    msg = "Loading checkpoint from {}".format(latest_checkpoint)
    ckpt.restore(latest_checkpoint)
  else:
    raise ValueError('No checkpointed model.')
  return mrsds_model, x_transition_networks, continuous_state_init_dist
