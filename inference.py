
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import layers, Input, models

from mrsds.utils_tensor import write_updates_to_tas, tensor_for_ta, normalize_logprob



class TransformerInferenceNetwork(tf.keras.Model):
  """
  Transformer Inference network for approx. posterior q(x[1:T] | y[1:T], u[1:T]).
  """

  def __init__(self, transformer, posterior_dist, latent_dims,
               embedding_nets=None):
    """
    Args:
      transformer: tf model, see networks.py
      posterior_dist: tf model, eg GaussianPosterior in inference.py
      latent_dims: list of latent dimensions per region
      embedding_nets: optional. tf model, see build_mr_embedding_net in inference.py
      stages: optional. trials x time x 1 tensor annotating the trial experiment stage.
      use_task_ids: flag. Use per trial task ids when training on multiple tasks.
    """
    super(TransformerInferenceNetwork, self).__init__()
    self.latent_dims = latent_dims
    self.transformer = transformer
    self.posterior_dist = posterior_dist
    if embedding_nets is None:
      self.embedding_nets = [lambda y: y] * num_days
    else:
      if type(embedding_nets) != list:
        embedding_nets = [embedding_nets]
      self.embedding_nets = embedding_nets

  @tf.function #(experimental_relax_shapes=True)
  def call(self, inputs, num_samples=1, dtype=tf.float32, random_seed=131,
           xmean=False, masks=None, embed_id=0):
    """
    Args:
      inputs: observations tensor, trials x time x neurons
      embed_id: index for multiday/animal stitching.
      cond_id:
      task_ids: trials x 1 tensor, when using multiple tasks.
      num_samples: number of posterior samples to take.
      xmean: flag to return the mean of q(x) instead of samples.
      masks: trials x time x neurons tensor, for cosmooth eval, dropout training
      history_inputs:

    Returns:
      x_sampled: samples x trials x time x num_latents tensor
        samples from approx. posterior. Note these can be xs or psis (in svae case)
      entropies: entropy of the approximate distribution q(x)
      log_probs: samples x trials x time x 1 tensor. Log prob under q(x) of samples
    """
    tf.random.set_seed(random_seed)

    batch_size, num_steps = tf.unstack(tf.shape(inputs[0])[:2])
    latent_dims = self.latent_dims
    ys, us = inputs

    # pass ys through day specific input embedding net
    # returns list of embeddings per region, which get concatenated
    embed_id = tf.convert_to_tensor(embed_id, dtype=tf.int32)
    embeds = {i: lambda: self.embedding_nets[i]([ys]) for i
              in range(len(self.embedding_nets))}
    embed = tf.switch_case(embed_id, embeds)
    ys_embedding = tf.concat(embed, axis=-1)

    print('ys embed', ys_embedding.shape)

    # -- Pass through transformer --

    # NOTE consider adding in animal and day id here too.
    # NOTE task ids will probably be missing time dim.
    h_inputs = [ys_embedding, us]
    hs = tf.concat(h_inputs, axis=-1)

    transformer_out = self.transformer(hs, mask=tf.cast(masks, tf.bool))
    xdist = self.posterior_dist(transformer_out)
    if xmean:
      x_sampled = xdist.mean()
    else:
      x_sampled = xdist.sample(num_samples, seed=random_seed)
    entropies = xdist.entropy()
    log_probs = xdist.log_prob(x_sampled)
    return x_sampled, entropies, log_probs


class TransformerInferenceNetworkXZ(tf.keras.Model):
  """Inference network for posterior q(z[1:T], psi[1:T] | y[1:T])."""

  def __init__(self, transformer, posterior_dist, latent_dims,
               num_states=1, embedding_nets=None):
    """
    Args:
      transformer: tf model, see networks.py
      posterior_dist: tf model, eg GaussianPosterior in inference.py
      latent_dims: list of latent dimensions per region
      num_states: number of discretes states z, gives dims of posterior q(z)
      embedding_nets: optional. tf model, see build_mr_embedding_net in inference.py
      stages: optional. trials x time x 1 tensor annotating the trial experiment stage.
      #use_task_ids: flag. Use per trial task ids when training on multiple tasks.
    """
    super(TransformerInferenceNetworkXZ, self).__init__()
    self.latent_dims = latent_dims
    self.transformer = transformer
    self.posterior_dist = posterior_dist
    if embedding_nets is None:
      self.embedding_nets = [lambda y: y] * num_days
    else:
      if type(embedding_nets) != list:
        embedding_nets = [embedding_nets]
      self.embedding_nets = embedding_nets
    self.num_states = num_states

  @tf.function #(experimental_relax_shapes=True)
  def call(self, inputs, num_samples=1, dtype=tf.float32, random_seed=131,
           xmean=False, masks=None, embed_id=0):
    """
    Args:
      inputs: observations tensor, trials x time x neurons
      embed_id: index for multiday/animal stitching.
      cond_id:
      task_ids: trials x 1 tensor, when using multiple tasks.
      num_samples: number of posterior samples to take.
      xmean: flag to return the mean of q(x) instead of samples.
      masks: trials x time x neurons tensor, for cosmooth eval, dropout training
      history_inputs:

    Returns:
      z_logprob: samples x trials x time x num_states tensor.
        for switching svae, log approx. posterior on discrete states q(z)
      psi_sampled: samples x trials x time x num_latents tensor
        samples from approx. posterior q(psi).
      entropies: entropy of the approximate distribution q(psi)
      log_probs: samples x trials x time x 1 tensor. Log prob under q(psi) of samples
    """
    tf.random.set_seed(random_seed)

    batch_size, num_steps = tf.unstack(tf.shape(inputs[0])[:2])
    latent_dims = self.latent_dims
    ys, us = inputs

    # pass ys through day specific input embedding net
    # returns list of embeddings per region, which get concatenated
    embed_id = tf.convert_to_tensor(embed_id, dtype=tf.int32)
    embeds = {i: lambda: self.embedding_nets[i]([ys]) for i
              in range(len(self.embedding_nets))}
    embed = tf.switch_case(embed_id, embeds)
    ys_embedding = tf.concat(embed, axis=-1)

    # -- Pass through transformer --
    hs = tf.concat([ys_embedding, us], axis=-1)
    transformer_out = self.transformer(hs, mask=tf.cast(masks, tf.bool))

    z_logprob = normalize_logprob(transformer_out[:,:,-self.num_states:], axis=-1)[0]
    psi_dist = self.posterior_dist(transformer_out[:,:,self.num_states:])
    if xmean:
      psi_sampled = psi_dist.mean()
    else:
      psi_sampled = psi_dist.sample(num_samples, seed=random_seed)
    entropies = psi_dist.entropy()
    log_probs = psi_dist.log_prob(psi_sampled)
    return z_logprob, psi_sampled, entropies, log_probs


class RnnInferenceNetwork(tf.keras.Model):
  """Inference network for posterior q(x[1:T] | y[1:T])."""

  def __init__(self, posterior_rnn, posterior_dist, latent_dims,
               embedding_nets=None, embedding_birnn=None, x0_post_dist=None,
               num_days=1, hist_tlen=1, hist_dim=1,
               xic_input_type='hs', xic_input_len=-1):
    """
    Construct an RnnInferenceNetwork instance.

    Args:
      posterior_rnn: RNN cell
        h[t] = f_RNN(h[t-1], x[t-1], u[t])
      posterior_dist: tfp distribution
        p(x[t] | h[t])
      latent_dims: int scalar
        Dimensions of the continuous latent x (across regions).
      embedding_net:
        an optional network to embed the observations `y[t]`.
        Default to `None`, in which case, no embedding is applied.
    """
    super(RnnInferenceNetwork, self).__init__()
    self.latent_dims = latent_dims
    self.posterior_rnn = posterior_rnn
    self.posterior_dist = posterior_dist
    self.num_days = num_days
    self.hist_tlen = hist_tlen
    self.hist_dims = hist_dim

    if embedding_nets is None:
      self.embedding_nets = [lambda y: y] * num_days
    else:
      if type(embedding_nets) != list:
        embedding_nets = [embedding_nets]
      self.embedding_nets = embedding_nets
    #assert(self.num_days == len(self.embedding_nets))
    self.embedding_birnn = embedding_birnn
    self.x0_post_dist = x0_post_dist
    self.xic_input_type = xic_input_type
    self.xic_input_len = xic_input_len

  def call(self, inputs, current_state=None, num_samples=1, embed_id=0,
           dtype=tf.float32, random_seed=0, masks=None, history_inputs=None):
    """
    Recursively sample x[t] ~ q(x[t]|h[t] = f_RNN(h[t-1], x[t-1], h[t]^b)).

    Args:
      inputs: List of tf.32 [batch_size, num_steps, obs_dim] ... also uncludes u.
        , whereeach observation should be flattened.
      num_samples: an `int` scalar for number of samples per time-step, for
        posterior inference networks, `x[i] ~ q(x[1:T] | y[1:T])`.
      dtype: The data type of input data.
      random_seed: an `Int` as the seed for random number generator.
      parallel_iters: a positive `Int` indicates the number of iterations
        allowed to run in parallel in `tf.while_loop`, where `tf.while_loop`
        defaults it to be 10.

    Returns:
      x_sampled: a float 3-D `Tensor` of size [num_samples, batch_size,
      num_steps, latent_dims], which stores the x_t sampled from posterior.
      entropies: a float 2-D `Tensor` of size [num_samples, batch_size,
      num_steps], which stores the entropies of posterior distributions.
      log_probs: a float 2-D `Tensor` of size [num_samples. batch_size,
      num_steps], which stores the log posterior probabilities.
    """
    batch_size, num_steps, num_neurons = tf.unstack(tf.shape(inputs[0])[:3])
    num_samples = tf.convert_to_tensor(num_samples, dtype_hint=tf.int32)
    latent_dims = self.latent_dims
    ys, us = inputs

    # Pass ys through day specific input embedding net
    # Returns list of embeddings per region
    ys_embedding = tf.concat(self.embedding_nets[embed_id]([ys]), axis=-1)

    # Shared birnn backbone to embed ys, us
    # NOTE dropped u from birnn for now because network was ignoring..
    if self.embedding_birnn is not None:
      hs = self.embedding_birnn(tf.concat([ys_embedding, us], axis=-1),
                                mask=tf.cast(masks, tf.bool))
    else:
      hs = tf.concat(ys_embedding, axis=-1)

    # -- Pass through causal RNN --

    ta_names = ["rnn_states", "xs", "entropies", "log_probs"]
    tas = [tf.TensorArray(tf.float32, num_steps, name=n) for n in ta_names]
    t0 = tf.constant(0, tf.int32)  # time/iter counter
    loopstate = namedtuple("LoopState", "rnn_state xs")

    # Set initial RNN state
    bs = batch_size * num_samples
    r0 = self.posterior_rnn.get_initial_state(batch_size=bs, dtype=dtype)
    rnn_classes = (layers.GRUCell, layers.SimpleRNNCell)
    if any(isinstance(self.posterior_rnn, _) for _ in rnn_classes):
      r0 = [r0]

    # Set inputs for initial latent state x0
    if history_inputs is None:
      history_inputs = tf.zeros([batch_size*num_samples, self.hist_dims])
      print('Using dummy history inputs')
    else: # tile to get multiple samples
     history_inputs = tf.tile(history_inputs, [num_samples,1])

    # Concat with first few timepoints of per region embeddings
    if self.xic_input_len == 0:
      ic_inputs = history_inputs
    else:
      if self.xic_input_type == 'hs':
        hs0 = tf.reshape(hs[:,:self.xic_input_len,:], [batch_size, -1])
      elif self.xic_input_type == 'yes':
        hs0 = tf.reshape(ys_embedding[:,:self.xic_input_len,:], [batch_size, -1])
      elif self.xic_input_type == 'ys':
        hs0 = tf.reshape(ys[:,:self.xic_input_len,:], [batch_size, -1])
      hs0_tiled = tf.tile(hs0, [num_samples,1])
      ic_inputs = tf.concat([hs0_tiled, history_inputs], axis=-1)

    # Sample initial state x0
    if self.x0_post_dist.nets[0] is None:
      x0 = tf.zeros([batch_size*num_samples, latent_dims], dtype=tf.float32)
    else:
      x0_dist = self.x0_post_dist(ic_inputs)
      x0 = x0_dist.sample(seed=random_seed)

    ls = loopstate(rnn_state=r0, xs=x0)
    init_state = (t0, ls, tas)

    def _cond(t, *unused_args):
      return t < num_steps

    def _step(t, loop_state, tas):
      """One step in tf.while_loop."""
      prev_xs = loop_state.xs
      prev_rnn_state = loop_state.rnn_state

      # Duplicate current us and embedded ys to sample multiple trajectories.
      current_us = tf.tile(us[:,t,:], [num_samples,1])
      current_hs = tf.tile(hs[:,t,:], [num_samples,1])

      # num_samples * BS, latent_dims+input_dim
      rnn_input = tf.concat([prev_xs, current_hs, current_us], axis=-1)
      rnn_out, rnn_state = self.posterior_rnn(inputs=rnn_input, states=prev_rnn_state)
      dist = self.posterior_dist(rnn_out)
      xs = dist.sample(seed=random_seed)

      # rnn_state is a list of [batch_size, rnn_hidden_dim].
      # After ta.stack(), will be [num_steps, {1 for GRU / 2 for LSTM}, batch, rnn_dim]
      tas_updates = [rnn_state, xs, dist.entropy(), dist.log_prob(xs)]
      tas = write_updates_to_tas(tas, t, tas_updates)
      ls = loopstate(rnn_state=rnn_state, xs=xs)
      return (t+1, ls, tas)

    _, _, tas_final = tf.while_loop(_cond, _step, init_state, parallel_iterations=1)
    x_sampled, entropies, log_probs = [tensor_for_ta(ta, swap_batch_time=True)
                                       for ta in tas_final[1:]]

    sizes = [num_samples, batch_size, num_steps]
    x_sampled = tf.reshape(x_sampled, [*sizes, latent_dims])
    entropies = tf.reshape(entropies, sizes)
    log_probs = tf.reshape(log_probs, sizes)
    return x_sampled, entropies, log_probs


class GaussianPosterior(tf.keras.Model):
  """
  Posterior  q(x[t]|r[t]) where r[t] is output of rnn.
  Note the function supports a list of functions for multiple days.
  """

  def __init__(self, nets, latent_dims, cov_mats=None,
               use_triangular_cov=True, use_trainable_cov=True,
               raw_sigma_bias=0.0, sigma_min=1e-5, sigma_scale=0.05,
               dtype=tf.float32, name="GaussianPosterior"):
    """
    TBD.

    Args:
      nets: list of `callable` networks taking RNN output continuous hidden
        states, `r[t]`, and returning the latent distribution `q(x[t] | r[t])`.
      latent_dims: an `int` scalar for dimension of latents `x`.
      cov_mat: an optional `float` Tensor for predefined covariance matrix.
        Default to `None`, in which case, a `cov` variable will be created.
      use_triangular_cov: a `bool` scalar indicating whether to use triangular
        covariance matrices and `tfp.distributions.MultivariateNormalTriL` for
        distribution. Otherwise, a diagonal covariance matrices and
        `tfp.distributions.MultivariateNormalDiag` will be used.
      use_trainable_cov: a `bool` scalar indicating whether the scale of
        the distribution is trainable. Default to True.
      raw_sigma_bias: a `float` scalar to be added to the raw sigma, which is
        standard deviation of the distribution. Default to `0.`.
      sigma_min: a `float` scalar for minimal level of sigma to prevent
        underflow. Default to `1e-5`.
      sigma_scale: a `float` scalar for scaling the sigma. Default to `0.05`.
        The above three arguments are used as
        `sigma_scale * max(softmax(raw_sigma + raw_sigma_bias), sigma_min))`.
      dtype: data type for variables within the scope. Default to `tf.float32`.
      name: a `str` to construct names of variables.
    """
    super(GaussianPosterior, self).__init__()
    if type(latent_dims) != list:
        latent_dims = [latent_dims]
    if type(nets) != list:
        nets = [nets]
    if cov_mats and type(cov_mats) != list:
        cov_mats = [cov_mats]
    self.latent_dims = latent_dims
    self.nets = nets
    self.num_days = len(nets)
    self.use_triangular_cov = use_triangular_cov

    if cov_mats:
      self.cov_mats = cov_mats

    else:
      cov_mats = []
      for latent_dim in self.latent_dims:
        if self.use_triangular_cov:
          shape = [int(latent_dim*(latent_dim+1)/2)]
        else:
          shape = [latent_dim]
        _ = tf.random.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=dtype)
        _ = tf.Variable(_, name="{}_cov".format(name), dtype=dtype, trainable=use_trainable_cov)
        if self.use_triangular_cov:
          _ = tfp.math.fill_triangular(_)
        cov_mat = tf.maximum(tf.nn.softmax(_ + raw_sigma_bias), sigma_min) * sigma_scale
        cov_mats.append(cov_mat)
      self.cov_mats = cov_mats

  def call(self, input_tensor, dtype=tf.float32, embed_id=0):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    mean_tensor = self.nets[embed_id](input_tensor)
    if self.use_triangular_cov:
      _ = tfd.MultivariateNormalTriL(loc=mean_tensor, scale_tril=self.cov_mats[embed_id])
    else:
      _ = tfd.MultivariateNormalDiag(loc=mean_tensor, scale_diag=self.cov_mats[embed_id])
    return tfp.experimental.as_composite(_)

  @property
  def output_event_dims(self):
    return self.latent_dims



class GaussianPosteriorDynamicCov(tf.keras.Model):
  """
  Posterior  q(x[t]|r[t]) where r[t] is output of rnn.
  Note the function supports a list of functions for multiple days.
  """

  def __init__(self, net, latent_dim, dynamic_cov=True, use_triangular_cov=True,
               raw_sigma_bias=0.0, sigma_min=1e-5, sigma_scale=0.05,
               dtype=tf.float32, name="GaussianPosterior"):
    """
    TBD.

    Args:
      use_triangular_cov: flag. a `bool` scalar indicating whether to use triangular
        covariance matrices and `tfp.distributions.MultivariateNormalTriL` for
        distribution. Otherwise, a diagonal covariance matrices and
        `tfp.distributions.MultivariateNormalDiag` will be used.
      use_trainable_cov: flag
      raw_sigma_bias: float added to sigma ( stdev of dist).
      sigma_min: float for min. sigma to prevent underflow
      sigma_scale: float for scaling sigma. Default to `0.05`.
    """
    super(GaussianPosteriorDynamicCov, self).__init__()

    self.latent_dim = latent_dim
    self.cov_dim = latent_dim
    if use_triangular_cov:
      self.cov_dim = int(self.latent_dim*(self.latent_dim+1)/2)
    self.dim_splits = [self.latent_dim, self.cov_dim]
    self.net = net
    #self.num_days = len(nets)
    self.use_triangular_cov = use_triangular_cov

  def call(self, input_tensor, dtype=tf.float32):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    mean_tensor, cov_tensor = tf.split(self.net(input_tensor),
                                       num_or_size_splits=self.dim_splits, axis=-1)
    if self.use_triangular_cov:
      _ = tfd.MultivariateNormalTriL(loc=mean_tensor, scale_tril=cov_tensor)
    else:
      _ = tfd.MultivariateNormalDiag(loc=mean_tensor, scale_diag=cov_tensor)
    return tfp.experimental.as_composite(_)

  @property
  def output_event_dims(self):
    return self.latent_dims


def build_mr_embedding_net(region_sizes, seq_len, num_neurons, dense_layers, input_len=2,
                           kernel_initializer="glorot_uniform",
                           bias_initializer="random_uniform"):
  """Builds an multiregion embedding network model for the inference network."""

  layer_kargs = {"kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer}

  # Split by region
  # Final split is padding, ignored if num_neurons > sum(region_sizes)
  num_regions = len(region_sizes)
  ys = Input(shape=(seq_len, num_neurons,))
  diff = num_neurons - np.sum(region_sizes)
  print('diff', diff, num_neurons, np.sum(region_sizes))
  splits = region_sizes
  if diff > 0:
    splits = np.append(region_sizes, diff)
  split = lambda y: tf.split(y, num_or_size_splits=splits, axis=-1)
  split_ys = layers.Lambda(split)(ys)

  # For each region, produce an embedding of the observations.
  ys_embedded = []
  for j in range(num_regions):
    yregion = split_ys[j]
    for lsize, activation in dense_layers:
      yregion = layers.Dense(lsize, activation, **layer_kargs)(yregion)
    ys_embedded.append(yregion)

  model = models.Model(inputs=[ys], outputs=ys_embedded)
  return model
