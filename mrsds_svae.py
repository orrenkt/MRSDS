"""
Multi-Region Switching Dynamical Systems (MR-SDS)
Structured inference.
"""
import yaml
import numpy as np
#import logging
#logging.getLogger("tensorflow").setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
import tensorflow as tf
tf.autograph.set_verbosity(0)
import tensorflow_probability as tfp
tfd = tfp.distributions
import time
from collections import namedtuple

from mrsds.generative_models import (ContinuousStateTransitionPsis2, ContinuousStateTransitionPsis,
                                     DiscreteStateTransition,
                                     GaussianEmissions, PoissonEmissions,
                                     build_mr_dynamics_net2, build_mr_emission_net,
                                     construct_x0_dist)
import mrsds.forward_backward as fb
from mrsds.inference import TransformerInferenceNetwork, GaussianPosterior, build_mr_embedding_net
from mrsds.utils_tensor import normalize_logprob, mask_logprob
from mrsds.utils_config import attr_dict_recursive, get_distribution_config
from mrsds.networks import build_birnn, build_dense_net, build_transformer
from mrsds.utils_tensor import write_updates_to_tas, tensor_for_ta


class MRSDS(tf.keras.Model):
  """
  """
  def __init__(self, x_transition_net, x_transition_net_time,
               emission_nets, inference_net,
               x0_distribution=None, continuous_state_dim=None,
               num_inputs=2, num_states=None, ux=True, dtype=tf.float32, psi_add=True):
    """
    TBD
    """
    super(MRSDS, self).__init__()

    self.x_tran = x_transition_net
    self.x_tran_time = x_transition_net_time
    self.y_emit = emission_nets
    self.inference_net = inference_net

    self.num_inputs = num_inputs

    self.num_states = num_states
    if num_states is None:
      self.num_states = 1

    self.x_dim = continuous_state_dim
    if continuous_state_dim is None:
      self.x_dim = self.x_tran.output_event_dims

    self.x0_dist = x0_distribution
    self.ux = ux
    self.psi_add = psi_add

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
    print('multistep')
    inputs, masks, time_masks = self._prep_inputs(ys, us, masks, dtype)
    num_samples = tf.convert_to_tensor(num_samples, dtype_hint=tf.int32)
    batch_size, num_steps = tf.unstack(tf.shape(inputs[0])[:2])

    print(inputs[0].shape, masks.shape)

    # Mask inputs for inference net.
    inputs_masked = [inputs[0] * masks, inputs[1]]

    start = time.time()
    # Sample continuous latent x ~ q(x[1:T] | y[1:T])
    psi_sampled, psi_entropy, psi_logprob = self.inference_net(inputs_masked,
                                                               num_samples=num_samples,
                                                               masks=time_masks,
                                                               random_seed=random_seed,
                                                               xmean=xmean)
    print('inf net time', time.time()-start)

    # Merge batch_size and num_samples dimensions (typically samples=1).
    shapes = [num_samples*batch_size, num_steps]
    x_dim = tf.shape(psi_sampled)[-1]
    psi_sampled = tf.reshape(psi_sampled, [*shapes, x_dim])
    psi_logprob = tf.squeeze(psi_logprob)

    # Mask the samples
    if masks is not None:
      psi_sampled *= time_masks[:,:,tf.newaxis]
      psi_entropy *= time_masks[:,:]

    print('inf time', time.time()-start)
    start = time.time()
    ic = (psi_sampled[:,0,:], psi_entropy[:,0], psi_logprob[:,0])
    x_sampled, x_entropy, log_probs = self.rollout(num_steps, psi_sampled, ic, inputs[1],
                                                   xmean=xmean)
    print('rollout time', time.time()-start)

    if masks is not None:
      x_sampled *= time_masks[:,:,tf.newaxis]
      x_entropy *= time_masks[:,:]

    # Reshape for samples
    inputs_ = [tf.tile(_, [num_samples,1,1]) for _ in inputs]

    # Based on sampled continuous states x_sampled, get discrete and continuous
    # state likelihoods.
    # log_a(j, k) = p(z[t]=j | z[t-1]=k, x[t-1], u[t])
    # log_b(k) = p(x[t] | x[t-1], u[t], z[t]=k)
    log_a = tf.zeros(1)
    start = time.time()
    log_px, xsample_prior, xmeans = self.compute_x_likelihood(x_sampled, inputs[1],
                                                              time_masks)
    print('xlik time', time.time()-start)
    logpx0 = tf.reduce_mean(log_px[:,0], axis=0)
    log_px = tf.reduce_sum(log_px, axis=-1)
    logpx_sum = tf.reduce_mean(log_px, axis=0)

    start = time.time()
    # Compute 0-step emissions and likelihoods p(y[1:T] | xinf[1:T])
    res = self.compute_y_likelihood(inputs_, x_sampled, time_masks, masks)
    y_dist, log_py, log_py_cosmooth = res
    log_py = tf.reduce_sum(log_py, axis=-1)
    ys_recon = y_dist.mean()
    if masks is not None:
      ys_recon *= time_masks[:,:,tf.newaxis]

    # 5 step overshoot
    w = [0.6, 0.2, 0.1] #, 0.05, 0.05] # last # baseline for 4 step.
    #w = [0.6, 0.2, 0.1, 0.05, 0.05] # last # baseline for 4 step.

    #w = [0.35, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]

    # 10 step overshoot (first timepoint is for inf)
    #w = [0.35, 0.15, 0.1, 0.075, 0.075, 0.075, 0.05, 0.05, 0.025, 0.025, 0.025]
    w = [0.45, 0.1, 0.1, 0.075, 0.075, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025]

    # 15 step overshoot
    #w = [0.3, 0.12, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 0.025, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.015]

    # 15 step overshoot focus on inf
    #w = [0.6, 0.05, 0.05, 0.05, 0.025, 0.025, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

    print('lxmeans', len(xmeans), len(w))
    ys_recons = []
    log_py_sum = log_py * w[0]
    for i in range(len(w)-1):
      res = self.compute_y_likelihood(inputs_, xmeans[i], time_masks, masks)
      y_dist_, log_py_, log_py_cosmooth_ = res
      #ys_recon_ = y_dist_.mean()
      #if masks is not None:
      #  ys_recon_ *= time_masks[:,:,tf.newaxis]
      # NOTE not saving these for now.
      #ys_recons.append(ys_recon_)
      #log_pys.append(tf.reduce_sum(log_py_, axis=-1))
      log_py_sum += tf.reduce_sum(log_py_, axis=-1) * w[1+i]

    print('ylik time', time.time()-start)

    # Compute ELBO, complete data likelihood and entropy
    log_p_xy = 100*beta*logpx0 + beta*log_px + log_py_sum #log_py
    elbo, log_pxy1T, log_qx = self.compute_objective(x_entropy, log_p_xy)

    print('logpx_sum', logpx_sum.shape, logpx_sum*beta)
    print('logpxy', log_p_xy.shape, tf.reduce_sum(log_p_xy))
    print('elbo', elbo.shape, elbo)

    # Reshape for samples if any
    _ = [batch_size, num_steps, -1]
    ys_recon = tf.reshape(ys_recon, _)
    #log_gamma1 = tf.reshape(log_gamma1, _)
    x_dim = tf.shape(x_sampled)[-1]
    x_sampled = tf.reshape(x_sampled, [*_, x_dim])
    #log_px_sum = tf.reduce_mean(tf.reduce_sum(log_px, axis=[1,2]), axis=0)

    # This got overridden by the new multistep logpy term
    log_py_sum = tf.reduce_mean(log_py_sum, axis=0)

    #@tf.autograph.experimental.do_not_convert
    def compute_smooth_penalty(xs, ws=[1,1,1],
                               #ws=[0.25,0.25,0.5],
                               time_masks=None):
      dx = xs[:,1:,:] - xs[:,:-1,:]
      dx2 = dx[:,1:,:] - dx[:,:-1,:]
      dx3 = dx2[:,1:,:] - dx2[:,:-1,:]
      if time_masks is not None:
        dx *= time_masks[:,1:]
        dx2 *= time_masks[:,2:]
        dx3 *= time_masks[:,3:]
      diffs_norm1 = tf.reduce_mean(tf.reduce_sum(tf.math.abs(dx), axis=[-1,-2])) #used to be [-1 only, kept time dim?
      diffs_norm2 = tf.reduce_mean(tf.reduce_sum(tf.math.abs(dx2), axis=[-1,-2]))
      diffs_norm3 = tf.reduce_mean(tf.reduce_sum(tf.math.abs(dx3), axis=[-1,-2]))
      #diffs_norm = diffs_norm1 * ws[0] + diffs_norm2 * ws[1] + diffs_norm3 * ws[2]
      diffs_norm = tf.squeeze(diffs_norm1) + tf.squeeze(diffs_norm2) + tf.squeeze(diffs_norm3)
      return diffs_norm

    # Inf smoothness
    diffs_norm = compute_smooth_penalty(tf.squeeze(x_sampled, axis=-2), time_masks)

    log_gamma1 = tf.zeros(1)
    log_b = tf.zeros(1)

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
      ("x_sampled", x_sampled),
      ("psi_sampled", psi_sampled),
      ("xsample_prior", xsample_prior),
      ("x_means", xmeans),
      ("elbo", elbo),
      #("ys_recons", ys_recons),
      #("x_means", x_means),
      ("diffs_norm", diffs_norm),
      ("sequence_likelihood", log_pxy1T),
      ("xt_entropy", log_qx),
      ("log_a", log_a),
      ("log_b", log_b)])

    return return_dict

  @tf.function
  def rollout(self, num_steps, psi_sampled, ic, us, random_seed=0,
              xmean=False):

    zeros_ = tf.zeros_like(psi_sampled[:,0,:])
    if self.psi_add:
      psi_sampled_ = tf.zeros_like(psi_sampled)
    else:
      psi_sampled_ = psi_sampled
    # Sample xs posterior given psis: q(x_1:T|y_1:T) = p(x_1|psi_1) \prod p(x_t|x_t-1, psi_t)
    ta_names = ["xs", "entropies", "log_probs"]
    tas = [tf.TensorArray(tf.float32, num_steps, name=n) for n in ta_names]
    t0 = tf.constant(1, tf.int32)  # time/iter counter
    loopstate = namedtuple("LoopState", "x")
    x0 = psi_sampled[:,0,:] + self.x0_dist.sample()
    psi0_entropy, psi0_logprob = ic[1:]
    ls = loopstate(x=x0)
    init_state = (t0, ls, tas)
    init_updates = (x0, psi0_entropy, psi0_logprob)
    tas = write_updates_to_tas(tas, 0, init_updates)

    def _cond(t, *unused_args):
      return t < num_steps

    def _step(t, loop_state, tas):
      x = loop_state.x
      u_t = us[:,t,:]
      psi_t = psi_sampled[:,t,:]
      inputs_ = [x, u_t, psi_t]
      if self.psi_add:
        inputs_[-1] = zeros_
        xdist = self.x_tran(*inputs_)
      else:
        xdist = self.x_tran(*inputs_)
      if xmean:
        x = xdist.mean()
      else:
        x = xdist.sample(seed=random_seed)
      if self.psi_add:
        x += psi_t

      tas_updates = [x, xdist.entropy(), xdist.log_prob(x)]
      tas = write_updates_to_tas(tas, t, tas_updates)
      ls = loopstate(x=x)
      return (t+1, ls, tas)

    _, _, tas_final = tf.while_loop(_cond, _step, init_state, parallel_iterations=1)
    x_sampled, entropies, log_probs = [tensor_for_ta(ta, swap_batch_time=True)
                                                     for ta in tas_final[:]]
    [_.close() for _ in tas]

    return x_sampled, entropies, log_probs

  def _prep_inputs(self, inputs, us, masks, dtype=tf.float32):
    """
    Read in masks and inputs.
    Mask is for specifying any padded / dropped neurons.
    Time mask is for masking timepoints only.
    """
    #inputs = tf.convert_to_tensor(inputs, dtype_hint=dtype, name="MRSDS_Input_Tensor")
    batch_size, num_steps = tf.unstack(tf.shape(inputs)[:2])
    if masks is not None:
      #masks = tf.convert_to_tensor(masks, dtype_hint=dtype, name="MRSDS_Mask_Tensor")
      # Add a time dim if not given
      if len(tf.unstack(tf.shape(masks))) == 2:
        masks = tf.expand_dims(masks, axis=1)
        masks = tf.repeat(masks, repeats=[num_steps], axis=1)
    else:
      masks = tf.ones_like(inputs, dtype=dtype)
      print('masks none. null mask:', inputs.shape, masks.shape)
    time_masks = tf.squeeze(masks[:,:,0])

    if us is not None:
      pass
      #us = tf.convert_to_tensor(us, dtype_hint=dtype, name="MRSDS_US_Tensor")
    else:
      us = tf.zeros([batch_size, num_steps, self.num_inputs], dtype=dtype)

    inputs = [inputs, us]
    return inputs, masks, time_masks

  def compute_objective(self, log_qx, log_p_xy):
    """
    Given all precalculated probabilities, return ELBO.
    """
    log_qx = tf.reduce_mean(tf.reduce_sum(log_qx, axis=1), axis=0)
    log_p_xy = tf.reduce_mean(log_p_xy, axis=0)
    elbo = log_p_xy + log_qx
    return elbo, log_p_xy, log_qx

  def compute_x_likelihood(self, x_sampled, us, time_masks, xsample=False):

    # --- Compute log p(x[t] | x[t-1], z[t], u[t]) ---

    # log_prob_x0 is [batch_size, num_states]
    x0_dist = self.x0_dist
    x_sampled0 = x_sampled[:,0,:]
    log_prob_x0 = x0_dist.log_prob(x_sampled[:,:1,:])

    # log_prob_xt is of shape [batch_size, num_steps, num_states]
    res = self.compute_x_transition_probs_multistep(x_sampled, log_prob_x0, us)
    log_prob_xt, xsample_prior, xmeans = res
    xt1_gen_dist = None

    # Mask out padded timepoints
    if time_masks is not None:
      if len(tf.unstack(tf.shape(log_prob_xt))) == 2:
        mask_ = time_masks
      else:
        mask_ = time_masks[:,:,tf.newaxis]
      log_prob_xt = mask_logprob(log_prob_xt, mask_, 'x')
      xsample_prior = mask_logprob(xsample_prior, time_masks[:,:,tf.newaxis], 'x')

    return log_prob_xt, xsample_prior, xmeans #xt1_gen_dist

  def compute_x_transition_probs_multistep(self, x_sampled, log_prob_x0, us=None):
    """
    p(x[t] | x[t-1], z[t]) transition.
    """
    if us is not None and self.ux:
      inputs_ = [x_sampled[:,:-1,:], us[:,1:,:]]
    else:
      nt, T = tf.unstack(tf.shape(x_sampled)[:2])
      inputs_ = [x_sampled[:,:-1,:], tf.zeros([nt,T-1,2])]

    xx = inputs_[0] #[:,:,tf.newaxis,:]
    uu = inputs_[1] #[:,:,tf.newaxis,:]
    psis = tf.zeros_like(xx)
    future_tensor = x_sampled[:,1:,:]
    start = time.time()
    dist = self.x_tran_time(xx,uu,psis,)
    xsample = dist.sample()
    log_probs = dist.log_prob(future_tensor)

    print('vec map xlik time', time.time()-start)
    log_prob_xt1 = tf.concat([log_prob_x0, log_probs], axis=1)
    xsample1 = tf.concat([x_sampled[:,:1,:], xsample], axis=1)

    # Higher order overshooting steps
    start = time.time()
    w = [0.45, 0.25, 0.2, 0.1]  # baseline for 4 step
    #w = [0.45, 0.2, 0.15, 0.1, 0.1]

    # 10 step overshoot
    w = [0.35, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]

    # 15 step overshoot
    #w = [0.3, 0.125, 0.1, 0.1, 0.075, 0.05, 0.05, 0.05, 0.025, 0.025, 0.02, 0.02, 0.02, 0.02, 0.02]

    xmean_prev = xsample1[:,:-1,:]
    xmeans = []
    log_probs = []
    xpred_dists = []
    for l in range(1,len(w)+1):

      xx = xmean_prev
      dist = self.x_tran_time(xx,uu,psis)
      xsample_ = dist.sample()
      log_probs_ = dist.log_prob(future_tensor)
      log_prob_xt = tf.concat([log_prob_x0, log_probs_], axis=1)
      xsample = tf.concat([x_sampled[:,:1,:], xsample_], axis=1)
      xmean_prev = xsample[:,:-1,:]

      xmeans.append(xsample)
      log_probs.append(log_prob_xt)

    # Average the log probs
    log_prob_xt = log_prob_xt1*w[0]
    for l in range(len(w)-1):
      log_prob_xt += log_probs[l] * w[l+1]

    print('xlik overshoot time', time.time()-start)
    return log_prob_xt, xsample1, xmeans

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


def build_model(model_dir, config_path, num_regions, num_dims,
                region_sizes, trial_length, num_states,
                name='mrsds', psi_add=True, zcond=False, random_seed=0):

  print('build model svae multitep mv')
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
  hidden_dim = np.sum(latent_region_sizes)
  num_neurons = np.sum(region_sizes)
  if hasattr(cfmxd, "single_region_latent"):
    if cfmxd.single_region_latent:
      latent_region_sizes = [num_dims*num_regions]

  listzip = lambda x: list(zip(*x))

  ## ----- Generative model -----

  # --- Continuous transitions ---

  xic_inf_net = None
  x0_dist = construct_x0_dist(
      latent_dims=hidden_dim,
      num_states=num_states,
      use_trainable_cov=True, #cfm.xic.trainable_cov,
      use_triangular_cov=True, #cfm.xic.triangular_cov,
      name=name+"_xic")

  print('x0 dist', x0_dist.parameters)
  print('x0 dist', x0_dist.variance)

  # Configuring p(x[t] | x[t-1], z[t])
  cxt = cfm.xtransition
  seq_len = np.max(trial_length)-1
  dynamics_layers = [[*cxt.dynamics.layers, hidden_dim/len(latent_region_sizes)],
                      cxt.dynamics.activations]
  num_inputs = cfd.inputs
  input_layers = [[*cxt.inputs.layers, latent_region_sizes[0]],
                  cxt.inputs.activations]
  inputs_ = [latent_region_sizes, seq_len,
             num_inputs,
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

  x_transition_nets = [build_mr_dynamics_net2(*inputs_, **cond_dict,
                                              seed=random_seed+i)
                       for i in range(num_states)]
  x_transition_nets, x_transition_nets_time = [[_] for _ in x_transition_nets[0]]
  x_transition = ContinuousStateTransitionPsis2(
    transition_mean_nets=x_transition_nets,
    distribution_dim=hidden_dim,
    num_states=num_states,
    use_triangular_cov=True, #cfg_xtr.use_triangular_cov,
    use_trainable_cov=cfg_xtr.use_trainable_cov,
    raw_sigma_bias=cfg_xtr.raw_sigma_bias,
    sigma_min=cfg_xtr.sigma_min,
    sigma_scale=cfg_xtr.sigma_scale,
    ux=cfm.ux, name=name+"_x_trans")
  x_transition_time = ContinuousStateTransitionPsis(
    transition_mean_nets=x_transition_nets_time,
    distribution_dim=hidden_dim,
    num_states=num_states,
    use_triangular_cov=True, #cfg_xtr.use_triangular_cov,
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
  emission_net = build_mr_emission_net(*inputs, xdropout=xdropout,
                                       layer_reg=layer_reg)

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
  print('transformer input shape', emb.layers[-1], num_inputs)
  print(emb.layers)
  print(embedding_net.output_shape, input_len)
  task_len = 0
  input_shape = (seq_len+1, emb.layers[-1]*num_regions + input_len + task_len,)

  print('input shape', input_shape, emb.layers[-1], num_regions, num_inputs)
  transformer = build_transformer(input_shape, hidden_dim, cfit.head_size, cfit.num_heads,
                                  cfit.ff_dim, cfit.num_transformer_blocks, cfit.mlp_units)
  print(transformer.summary())
  posterior_net = TransformerInferenceNetwork(
    transformer=transformer,
    posterior_dist=posterior_distribution,
    latent_dims=hidden_dim, embedding_nets=embedding_net,
    )

  # ----- Build MRSDS model -----

  mrsds_model = MRSDS(
    x_transition_net=x_transition,
    x_transition_net_time=x_transition_time,
    emission_nets=emission_nets,
    inference_net=posterior_net,
    x0_distribution=x0_dist,
    continuous_state_dim=None,
    num_states=None,
    num_inputs=num_inputs,
    ux=cfm.ux, psi_add=psi_add)

  shapes = (cft.batch_size, trial_length, num_neurons)
  mrsds_model.build(input_shape=shapes)
  summary = mrsds_model.summary()
  print(summary)

  xic_nets = None
  return mrsds_model, (x_transition_nets, x_transition_nets_time), xic_nets


def load_model(model_dir, config_path, num_regions, num_dims,
               region_sizes, trial_length, num_states, num_inputs=0,
               load=True, psi_add=True):

  from mrsds.mrsds_svae import build_model
  from mrsds.utils_config import load_yaml_config

  print('svae multistep')

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
                                             num_states=num_states, zcond=zcond,
                                             psi_add=psi_add)
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
