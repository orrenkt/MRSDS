
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import layers, Input, models, regularizers

from mrsds.utils_tensor import normalize_logprob, clamp_probs
from mrsds.utils_tensor import write_updates_to_tas, tensor_for_ta


def construct_x0_dist(latent_dims, num_states, use_trainable_cov=True,
                      use_triangular_cov=False, raw_sigma_bias=0.0,
                      sigma_min=1e-5, sigma_scale=0.3,
                      dtype=tf.float32, name="x0"):
  """
  Construct continuous initial state distribution p(x[0]).

  Args:
    latent_dims: int scalar, number of dimensions of latent x
    num_states: int scalar, number of discrete states z (k)
    use_trainable_cov: flag for traininable cov, default True.
    use_triangular_cov: flag for using tfp.distributions.MultivariateNormalTriL
      instead of default tfp.distributions.MultivariateNormalDiag
    raw_sigma_bias: float scalar added to raw sigma
    sigma_min: float scalar to precent underflow
    sigma_scale: float scalar for scaling sigma

  Returns:
    return_dist: tfp.distribution for p(x0)
  """
  glorot_initializer = tf.keras.initializers.GlorotUniform()
  init = glorot_initializer(shape=[num_states, latent_dims], dtype=dtype)
  x0_mean = tf.Variable(initial_value=init, name="{}_mean".format(name))

  if use_triangular_cov:
    shape = [int(latent_dims * (latent_dims + 1) / 2)]
    _ = tf.Variable(initial_value=glorot_initializer(shape=shape, dtype=dtype),
                    name="{}_scale".format(name), trainable=use_trainable_cov)
    _ = tfp.math.fill_triangular(_)
    x0_scale = tf.maximum(tf.nn.softmax(_ + raw_sigma_bias), sigma_min) * sigma_scale
    return_dist = tfd.Independent(
        distribution=tfd.MultivariateNormalTriL(loc=x0_mean, scale_tril=x0_scale),
        reinterpreted_batch_ndims=0)

  else:
    _ = tf.Variable(initial_value=glorot_initializer(shape=[latent_dims], dtype=dtype),
                    name="{}_scale".format(name), trainable=use_trainable_cov)
    x0_scale = tf.maximum(tf.nn.softmax(_ + raw_sigma_bias), sigma_min) * sigma_scale
    return_dist = tfd.Independent(
        distribution=tfd.MultivariateNormalDiag(loc=x0_mean, scale_diag=x0_scale),
        reinterpreted_batch_ndims=0)

  return tfp.experimental.as_composite(return_dist)


class ContinuousStateTransition(tf.keras.Model):
  """
  Dynamics transition for mean field p(x[t] | x[t-1], z[t]).
  """

  def __init__(self, transition_mean_nets, distribution_dim,
               num_states=1, use_triangular_cov=False,
               use_trainable_cov=True, ux=True,
               raw_sigma_bias=0.0, sigma_min=1e-5, sigma_scale=0.05,
               dtype=tf.float32, name="ContinuousStateTransition"):
    """
    Args:
      transition_mean_nets: list of dynamics networks
      distribution_dim: int scalar, dimension of latent x
      num_states: int scalar, number of discrete states z
      use_trainable_cov: flag for traininable cov, default True.
      use_triangular_cov: flag for using tfp.distributions.MultivariateNormalTriL
        instead of default tfp.distributions.MultivariateNormalDiag
      raw_sigma_bias: float scalar added to raw sigma
      sigma_min: float scalar to precent underflow
      sigma_scale: float scalar for scaling sigma
    """
    super(ContinuousStateTransition, self).__init__()

    assertion_str = "Wrong number of networks"
    assert len(transition_mean_nets) == num_states, assertion_str
    self.x_trans_nets = transition_mean_nets
    self.num_states = num_states
    self.use_triangular_cov = use_triangular_cov
    self.distribution_dim = distribution_dim

    if self.use_triangular_cov:
      shape = int(self.distribution_dim * (self.distribution_dim + 1) / 2)
    else:
      shape = self.distribution_dim

    _ = tf.random.uniform(shape=[shape], minval=0., maxval=1., dtype=dtype)
    _ = tf.Variable(_, name="{}_cov".format(name), dtype=dtype, trainable=use_trainable_cov)
    if self.use_triangular_cov:
      _ = tfp.math.fill_triangular(_)
    self.cov_mat = tf.maximum(tf.nn.softmax(_ + raw_sigma_bias), sigma_min) * sigma_scale
    self.ux = ux

    # TODO: want different cov mats per dynamics model??

  def call(self, input_tensor, us_tensor, dtype=tf.float32):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    if len(input_tensor.shape) == 2:
      batch_size, dist_dim = tf.unstack(tf.shape(input_tensor))
    else:
      batch_size, num_steps, dist_dim = tf.unstack(tf.shape(input_tensor))

    # NOTE could also just zero out us instead.
    inputs = [input_tensor]
    if us_tensor is not None and self.ux:
      us_tensor = tf.convert_to_tensor(us_tensor, dtype_hint=dtype)
      if len(input_tensor) > len(us_tensor.shape):  # NOTE catch for x0 model
        us_tensor = tf.squeeze(us_tensor)
      inputs.append(us_tensor)
    # NOTE DEBUGGING recent addition of this
    else:
      #print('Using zeros for us')
      inputs.append(tf.zeros((batch_size, num_steps, 2)))
      #print('no us input to xtransition.')

    # The shape of the mean_tensor after tf.stack is
    # [num_states, batch_size, num_steps, distribution_dim]
    _ = tf.stack([x_net(inputs) for x_net in self.x_trans_nets])
    if len(input_tensor.shape) == 2:
      _ = tf.squeeze(_)
      mean_tensor = tf.squeeze(tf.reshape(_, [batch_size, self.num_states, dist_dim]))
    else:
      _ = tf.transpose(_, [1,2,0,3])
      mean_tensor = tf.reshape(_, [batch_size, num_steps, self.num_states, dist_dim])

    if self.use_triangular_cov:
      output_dist = tfd.MultivariateNormalTriL(loc=mean_tensor, scale_tril=self.cov_mat)
    else:
      output_dist = tfd.MultivariateNormalDiag(loc=mean_tensor, scale_diag=self.cov_mat)
    return tfp.experimental.as_composite(output_dist)

  @property
  def output_event_dims(self):
    return self.distribution_dim


class ContinuousStateTransitionZCond(tf.keras.Model):
  """
  Dynamics transition for mean field p(x[t] | x[t-1], z[t]).
  Note that in this class a single dynamics network is used to parameterize f_z,
  with z as a conditioning input.
  """

  def __init__(self, transition_mean_net, distribution_dim,
               num_states=1, use_triangular_cov=False,
               use_trainable_cov=True, raw_sigma_bias=0.0, sigma_min=1e-5,
               sigma_scale=0.05, ux=True, dtype=tf.float32,
               name="ContinuousStateTransition"):
    """
    Args:
      transition_mean_net: single dynamics networks, conditioned on z
      distribution_dim: int scalar, dimension of latent x
      num_states: int scalar, number of discrete states z
      use_trainable_cov: flag for traininable cov, default True.
      use_triangular_cov: flag for using tfp.distributions.MultivariateNormalTriL
        instead of default tfp.distributions.MultivariateNormalDiag
      raw_sigma_bias: float scalar added to raw sigma
      sigma_min: float scalar to precent underflow
      sigma_scale: float scalar for scaling sigma
    """
    super(ContinuousStateTransitionZCond, self).__init__()

    self.xtran_net = transition_mean_net
    self.num_states = num_states
    self.use_triangular_cov = use_triangular_cov
    self.distribution_dim = distribution_dim

    self.zs_onehot = []
    for i in range(num_states):
        z_onehot = np.zeros([1,1,num_states])
        z_onehot[:,:,i] = 1.0
        self.zs_onehot.append(tf.convert_to_tensor(z_onehot, dtype_hint=dtype))

    if self.use_triangular_cov:
      shape = int(self.distribution_dim * (self.distribution_dim + 1) / 2)
    else:
      shape = self.distribution_dim

    _ = tf.random.uniform(shape=[shape], minval=0., maxval=1., dtype=dtype)
    _ = tf.Variable(_, name="{}_cov".format(name), dtype=dtype, trainable=use_trainable_cov)
    if self.use_triangular_cov:
      _ = tfp.math.fill_triangular(_)
    self.cov_mat = tf.maximum(tf.nn.softmax(_ + raw_sigma_bias), sigma_min) * sigma_scale

    self.ux = ux

    # TODO: want different cov mats per dynamics model??

  def call(self, input_tensor, us_tensor, dtype=tf.float32):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    if len(input_tensor.shape) == 2:
      batch_size, dist_dim = tf.unstack(tf.shape(input_tensor))
    else:
      batch_size, num_steps, dist_dim = tf.unstack(tf.shape(input_tensor))

    # The shape of the mean_tensor after tf.stack is
    # [num_states, batch_size, num_steps, distribution_dim]
    inputs = [input_tensor]
    if us_tensor is not None and self.ux:
      us_tensor = tf.convert_to_tensor(us_tensor, dtype_hint=dtype)
      if len(input_tensor) > len(us_tensor.shape):  # NOTE catch for x0 model
        us_tensor = tf.squeeze(us_tensor)
      inputs.append(us_tensor)
    else:
      print('no us input to xtransition.')

    # Pass in discrete state and append calls
    _ = tf.stack([self.xtran_net([*inputs, tf.tile(z,[batch_size,num_steps,1])])
                  for z in self.zs_onehot])
    if len(input_tensor.shape) == 2:
      _ = tf.squeeze(_)
      mean_tensor = tf.squeeze(tf.reshape(_, [batch_size, self.num_states, dist_dim]))
    else:
      _ = tf.transpose(_, [1,2,0,3])
      mean_tensor = tf.reshape(_, [batch_size, num_steps, self.num_states, dist_dim])

    if self.use_triangular_cov:
      output_dist = tfd.MultivariateNormalTriL(loc=mean_tensor, scale_tril=self.cov_mat)
    else:
      output_dist = tfd.MultivariateNormalDiag(loc=mean_tensor, scale_diag=self.cov_mat)
    return tfp.experimental.as_composite(output_dist)

  @property
  def output_event_dims(self):
    return self.distribution_dim


class ContinuousStateTransitionPsis(tf.keras.Model):
  """
  Dynamics transition for svae p(x[t] | x[t-1], z[t], psi[t]).
  By default (when not running inference), psi = 0
  """

  def __init__(self, transition_mean_nets, distribution_dim,
               num_states=1, use_triangular_cov=True,
               use_trainable_cov=True, ux=True,
               raw_sigma_bias=0.0, sigma_min=1e-5, sigma_scale=0.05,
               dtype=tf.float32, name="ContinuousStateTransition"):
    """
    Args:
      transition_mean_nets: list of dynamics networks
      distribution_dim: int scalar, dimension of latent x
      num_states: int scalar, number of discrete states z
      use_trainable_cov: flag for traininable cov, default True.
      use_triangular_cov: flag for using tfp.distributions.MultivariateNormalTriL
        instead of default tfp.distributions.MultivariateNormalDiag
      raw_sigma_bias: float scalar added to raw sigma
      sigma_min: float scalar to precent underflow
      sigma_scale: float scalar for scaling sigma
    """
    super(ContinuousStateTransitionPsis, self).__init__()

    assertion_str = "Wrong number of networks"
    assert len(transition_mean_nets) == num_states, assertion_str
    self.x_trans_nets = transition_mean_nets
    self.num_states = num_states
    self.use_triangular_cov = use_triangular_cov
    self.distribution_dim = distribution_dim

    if self.use_triangular_cov:
      shape = int(self.distribution_dim * (self.distribution_dim + 1) / 2)
    else:
      shape = self.distribution_dim

    _ = tf.random.uniform(shape=[shape], minval=0., maxval=1., dtype=dtype)
    _ = tf.Variable(_, name="{}_cov".format(name), dtype=dtype, trainable=use_trainable_cov)
    if self.use_triangular_cov:
      _ = tfp.math.fill_triangular(_)
    self.cov_mat = tf.maximum(tf.nn.softmax(_ + raw_sigma_bias), sigma_min) * sigma_scale

    self.ux = ux

    # TODO: want different cov mats per dynamics model??

  def call(self, input_tensor, us_tensor, psis_tensor, dtype=tf.float32):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    num_steps = 1
    if len(input_tensor.shape) == 2:
      batch_size, dist_dim = tf.unstack(tf.shape(input_tensor))
    else:
      batch_size, num_steps, dist_dim = tf.unstack(tf.shape(input_tensor))

    # NOTE could also just zero out us instead.
    inputs = [input_tensor]
    if us_tensor is not None and self.ux:
      us_tensor = tf.convert_to_tensor(us_tensor, dtype_hint=dtype)
      if len(input_tensor.shape) > len(us_tensor.shape):  # NOTE catch for x0 model
        us_tensor = tf.squeeze(us_tensor)
      inputs.append(us_tensor)
    # NOTE DEBUGGING recent addition of this
    else:
      #print('Using zeros for us')
      if num_steps == 1:
        inputs.append(tf.zeros((batch_size, 2)))
      else:
        inputs.append(tf.zeros((batch_size, num_steps, 2)))
      #print('no us input to xtransition.')

    inputs.append(psis_tensor)

    # The shape of the mean_tensor after tf.stack is
    # [num_states, batch_size, num_steps, distribution_dim]
    _ = tf.stack([x_net(inputs) for x_net in self.x_trans_nets])
    if len(input_tensor.shape) == 2:
      _ = tf.squeeze(_)
      mean_tensor = tf.reshape(_, [batch_size, self.num_states, dist_dim])
      if self.num_states == 1:
        mean_tensor = tf.squeeze(mean_tensor, axis=1)
    else:
      _ = tf.transpose(_, [1,2,0,3])
      mean_tensor = tf.reshape(_, [batch_size, num_steps, self.num_states, dist_dim])
      if self.num_states == 1:
        mean_tensor = tf.squeeze(mean_tensor, axis=[2])

    if self.use_triangular_cov:
      output_dist = tfd.MultivariateNormalTriL(loc=mean_tensor, scale_tril=self.cov_mat)
    else:
      output_dist = tfd.MultivariateNormalDiag(loc=mean_tensor, scale_diag=self.cov_mat)
    return tfp.experimental.as_composite(output_dist)

  @property
  def output_event_dims(self):
    return self.distribution_dim


class ContinuousStateTransitionPsisZCond(tf.keras.Model):
  """
  Dynamics transition for svae p(x[t] | x[t-1], z[t], psi[t]).
  By default (when not running inference), psi = 0
  Note that in this class a single dynamics network is used to parameterize f_z,
  with z as a conditioning input.
  """

  def __init__(self, transition_mean_net, distribution_dim,
               num_states=1, use_triangular_cov=True,
               use_trainable_cov=True, ux=True,
               raw_sigma_bias=0.0, sigma_min=1e-5, sigma_scale=0.05,
               dtype=tf.float32, name="ContinuousStateTransition"):
    """
      transition_mean_net: single dynamics networks, conditioned on z
      distribution_dim: int scalar, dimension of latent x
      num_states: int scalar, number of discrete states z
      use_trainable_cov: flag for traininable cov, default true.
      use_triangular_cov: flag for using tfp.distributions.multivariatenormaltril
        instead of default tfp.distributions.multivariatenormaldiag
      raw_sigma_bias: float scalar added to raw sigma
      sigma_min: float scalar to precent underflow
      sigma_scale: float scalar for scaling sigma
    """
    super(ContinuousStateTransitionPsisZCond, self).__init__()

    assertion_str = "Wrong number of networks"
    self.xtran_net = transition_mean_net
    self.num_states = num_states
    self.use_triangular_cov = use_triangular_cov
    self.distribution_dim = distribution_dim

    self.num_steps = transition_mean_net.input_shape[1]

    self.zs_onehot = []
    for i in range(num_states):
        if self.num_steps == 1:
          z_onehot = np.zeros([1,num_states])
        else:
          z_onehot = np.zeros([1,1,num_states])
        z_onehot[:,:,i] = 1.0
        self.zs_onehot.append(tf.convert_to_tensor(z_onehot, dtype_hint=dtype))

    if self.use_triangular_cov:
      shape = int(self.distribution_dim * (self.distribution_dim + 1) / 2)
    else:
      shape = self.distribution_dim

    _ = tf.random.uniform(shape=[shape], minval=0., maxval=1., dtype=dtype)
    _ = tf.Variable(_, name="{}_cov".format(name), dtype=dtype, trainable=use_trainable_cov)
    if self.use_triangular_cov:
      _ = tfp.math.fill_triangular(_)
    self.cov_mat = tf.maximum(tf.nn.softmax(_ + raw_sigma_bias), sigma_min) * sigma_scale

    self.ux = ux

    # TODO: want different cov mats per dynamics model??

  def call(self, input_tensor, us_tensor, psis_tensor, dtype=tf.float32):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    if len(input_tensor.shape) == 2:
      batch_size, dist_dim = tf.unstack(tf.shape(input_tensor))
      num_steps = 1
    else:
      batch_size, num_steps, dist_dim = tf.unstack(tf.shape(input_tensor))

    # NOTE could also just zero out us instead.
    inputs = [input_tensor]
    if us_tensor is not None and self.ux:
      us_tensor = tf.convert_to_tensor(us_tensor, dtype_hint=dtype)
      if len(input_tensor.shape) > len(us_tensor.shape):  # NOTE catch for x0 model
        us_tensor = tf.squeeze(us_tensor)
      inputs.append(us_tensor)
    # NOTE DEBUGGING recent addition of this
    else:
      if len(input_tensor.shape) == 2:
        inputs.append(tf.zeros((batch_size, 2)))
      else:
        inputs.append(tf.zeros((batch_size, num_steps, 2)))

    inputs.append(psis_tensor)

    # The shape of the mean_tensor after tf.stack is
    # [num_states, batch_size, num_steps, distribution_dim]
    #_ = tf.stack([x_net(inputs) for x_net in self.x_trans_nets])
    # Pass in discrete state and append calls

    if num_steps == 1:
      _ = tf.stack([self.xtran_net([*inputs, tf.squeeze(tf.tile(z,[batch_size,1,1]))])
                    for z in self.zs_onehot])
      if len(input_tensor.shape) == 2:
        _ = tf.squeeze(_)
        mean_tensor = tf.squeeze(tf.reshape(_, [batch_size, self.num_states, dist_dim]))
      else:
        _ = tf.transpose(_, [1,2,0,3])
        mean_tensor = tf.squeeze(tf.reshape(_, [batch_size, num_steps, self.num_states, dist_dim]))

    else:
      _ = tf.stack([self.xtran_net([*inputs, tf.squeeze(tf.tile(z,[batch_size,num_steps,1]))])
                  for z in self.zs_onehot])
      if len(input_tensor.shape) == 2:
        _ = tf.squeeze(_)
        mean_tensor = tf.squeeze(tf.reshape(_, [batch_size, self.num_states, dist_dim]))
      else:
        _ = tf.transpose(_, [1,2,0,3])
        mean_tensor = tf.squeeze(tf.reshape(_, [batch_size, num_steps, self.num_states, dist_dim]))

    if self.use_triangular_cov:
      output_dist = tfd.MultivariateNormalTriL(loc=mean_tensor, scale_tril=self.cov_mat)
    else:
      output_dist = tfd.MultivariateNormalDiag(loc=mean_tensor, scale_diag=self.cov_mat)
    return tfp.experimental.as_composite(output_dist)

  @property
  def output_event_dims(self):
    return self.distribution_dim


class DiscreteStateTransition(tf.keras.Model):
  """
  Discrete state transition p(z[t] | z[t-1], x[t-1], u[t-1]).
  Input effect (us) is optional.
  """

  def __init__(self, transition_net, num_states):
    """
    Args:
      transition_net: transition dynamics network
      num_states: int scalar, number of discrete states z (k)
    """
    super(DiscreteStateTransition, self).__init__()
    self.dense_net = transition_net
    self.num_states = num_states

  def call(self, input_tensor, dtype=tf.float32):
    """Returns transition tensor."""
    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    batch_size, num_steps = tf.unstack(tf.shape(input_tensor)[:2])
    shape = (batch_size, num_steps, self.num_states, self.num_states)
    return tf.reshape(self.dense_net(input_tensor), shape)

  @property
  def output_event_dims(self):
    return self.num_states


class GaussianEmissions(tf.keras.Model):
  """Emission model p(y[t] | x[t])."""

  def __init__(self, emission_mean_nets, observation_dims,
               use_triangular_cov=True, use_trainable_cov=True,
               raw_sigma_bias=0.0, sigma_min=1e-5, sigma_scale=0.05,
               dtype=tf.float32, name="GaussianEmissions"):
    """
    Args:
      emission_mean_nets: list of emissions networks, one per region
      observation_dims: list of ints, number of neurons per region
      use_trainable_cov: flag for traininable cov, default true.
      use_triangular_cov: flag for using tfp.distributions.multivariatenormaltril
        instead of default tfp.distributions.multivariatenormaldiag
      raw_sigma_bias: float scalar added to raw sigma
      sigma_min: float scalar to precent underflow
      sigma_scale: float scalar for scaling sigma
    """
    super(GaussianEmissions, self).__init__()
    if type(observation_dims) != list:
        observation_dims = [observation_dims]
    if type(emission_mean_nets) != list:
        emission_mean_nets = [emission_mean_nets]
    self.obs_dims = observation_dims
    self.y_emission_nets = emission_mean_nets
    self.num_embeds = len(emission_mean_nets)
    self.use_triangular_cov = use_triangular_cov

    cov_mats = []
    obs_dim = self.obs_dims[0]
    for i in range(self.num_embeds):
      if self.use_triangular_cov:
        shape = [int(obs_dim*(obs_dim+1)/2)]
      else:
        shape = [obs_dim]
      _ = tf.random.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=dtype)
      _ = tf.Variable(_, name="{}_cov".format(name), dtype=dtype, trainable=use_trainable_cov)
      if self.use_triangular_cov:
        _ = tfp.math.fill_triangular(_)
      cov_mat = tf.maximum(tf.nn.softmax(_ + raw_sigma_bias), sigma_min) * sigma_scale
      cov_mats.append(cov_mat)
    self.cov_mats = cov_mats

  def call(self, input_tensor, dtype=tf.float32, embed_id=0, cond_id=0):

    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    batch_size, num_steps = tf.unstack(tf.shape(input_tensor)[:2])
    cond_id = tf.convert_to_tensor([cond_id], dtype_hint=dtype)
    cond_id_tensor = tf.zeros([batch_size, num_steps, 1], dtype=dtype) + cond_id
    mean_tensor = self.y_emission_nets[embed_id]([input_tensor])
    if self.use_triangular_cov:
      _ = tfd.MultivariateNormalTriL(loc=mean_tensor, scale_tril=self.cov_mats[embed_id])
    else:
      _ = tfd.MultivariateNormalDiag(loc=mean_tensor, scale_diag=self.cov_mats[embed_id])
    return tfp.experimental.as_composite(_)

  @property
  def output_event_dims(self):
    return self.obs_dims


class PoissonEmissions(tf.keras.Model):
  """
  Poisson Emission model p(y[t] | x[t]).
  """
  def __init__(self, emission_rate_nets, observation_dims,
               dtype=tf.float32, name="PoissonEmissions"):
    """
    Constructor for Poisson emissions class.
    """
    super(PoissonEmissions, self).__init__()
    if type(observation_dims) != list:
        observation_dims = [observation_dims]
    if type(emission_rate_nets) != list:
        emission_rate_nets = [emission_rate_nets]
    self.obs_dims = observation_dims
    self.y_emission_nets = emission_rate_nets
    self.num_days = len(emission_rate_nets)

  def call(self, input_tensor, dtype=tf.float32, embed_id=0, cond_id=0):
    input_tensor = tf.convert_to_tensor(input_tensor, dtype_hint=dtype)
    batch_size, num_steps = tf.unstack(tf.shape(input_tensor)[:2])
    rate_tensor = self.y_emission_nets[embed_id](input_tensor)
    _ = tfd.Poisson(rate=rate_tensor)
    return tfp.experimental.as_composite(_)

  @property
  def output_event_dims(self):
    return self.obs_dims


def build_mr_dynamics_net2(latent_region_sizes, seq_len, input_len,
                          dynamics_layers, input_layers=None, input_regions='all',
                          communication_type="additive",
                          input_type="additive", train_u=True,
                          input_func_type='seperate', # alt: together
                          kernel_initializer="glorot_uniform",
                          bias_initializer="random_uniform", zcond=False,
                          num_states=1, xdropout=False, dropout_rate=0.2, layer_reg=False,
                          seed=0, step_view=False):
  """Helper function for building an additive multi-region multi-layer network."""

  # Post iclr
  np.random.seed(seed)
  tf.random.set_seed(seed)

  layer_kargs = {"kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer}

  if layer_reg:
    layer_kargs['kernel_regularizer'] = regularizers.L2(1e-4)
    layer_kargs['bias_regularizer'] = regularizers.L2(1e-4)
    layer_kargs['activity_regularizer'] = regularizers.L2(1e-5)

  # -- Inputs --

  # Split latents by region
  num_regions = len(latent_region_sizes)
  xs1 = tf.keras.Input(shape=(np.sum(latent_region_sizes),))
  xs = tf.keras.Input(shape=(seq_len, np.sum(latent_region_sizes),))
  split = lambda x: tf.split(x, num_or_size_splits=latent_region_sizes, axis=-1)
  split_xs1 = layers.Lambda(split)(xs1)
  split_xs = layers.Lambda(split)(xs)

  if input_layers is not None:
    us1 = Input(shape=(input_len,))
    us = Input(shape=(seq_len, input_len,))
    inputs1 = [xs1, us1]
    split_us1 = [us1]
    inputs = [xs, us]
    split_us = [us]
    if input_func_type == "seperate":
      split = lambda u: tf.split(u, num_or_size_splits=input_len, axis=-1)
      split_us1 = layers.Lambda(split)(us1)
      split_us = layers.Lambda(split)(us)
  else:
    inputs1 = [xs1]
    inputs = [xs]

  split = lambda x: tf.split(x, num_or_size_splits=latent_region_sizes, axis=-1)
  psis1 = tf.keras.Input(shape=(np.sum(latent_region_sizes),))
  psis = tf.keras.Input(shape=(seq_len, np.sum(latent_region_sizes),))
  split_psis1 = layers.Lambda(split)(psis1)
  split_psis = layers.Lambda(split)(psis)
  inputs1.append(psis1)
  inputs.append(psis)

  # NOTE could actually make this always conditioned, just pass in 0 if no zcond.
  if zcond:
    z1 = tf.keras.Input(shape=(num_states,))
    inputs1 = [*inputs1, z1]
    z = tf.keras.Input(shape=(seq_len, num_states,))
    inputs = [*inputs, z]

  # -- Dynamics --

  num_layers = len(dynamics_layers)
  num_input_layers = len(input_layers)

  xs_next1 = []
  xs_next = []
  for i in range(num_regions):

    xlocal1 = split_xs1[i]
    xlocal = split_xs[i]

    # Additive local dynamics and communication from each region
    x_next1 = []
    x_next = []
    for j in range(num_regions):
      xregion1 = split_xs1[j]
      psi_region1 = split_psis1[j]
      xregion = split_xs[j]
      psi_region = split_psis[j]
      for l, (lsize, activation) in enumerate(dynamics_layers):
        if l == num_layers-1: # final layer
          name = "{}-{}".format(i, j)
        else:
          name = None
        if j == i:
          if zcond:
            xregion1 = layers.concatenate([xregion1, z1])
            xregion = layers.concatenate([xregion, z])
          _1 = xregion1
          _ = xregion
        else:
          if communication_type == "additive-linear":
            print('linear comms', l)
            #if l < num_layers-1:  # for keeping only final layer
            #  continue
            activation = None
          _1 = xregion1
          _ = xregion
          if communication_type == "additive-nonlinear": # so we set only comms to nonlinear
            print('nolinear comms', l)
            #if l < num_layers-1:  # for keeping only final layer
            #  continue
            activation = 'relu'
          _1 = xregion1
          _ = xregion
          if communication_type == "conditional":
            _1 = layers.concatenate([xregion1, xlocal1])
            _ = layers.concatenate([xregion, xlocal])
          if zcond:
            _1 = layers.concatenate([_1, z1])
            _ = layers.concatenate([_, z])

        if l == 0:
          _1 = layers.concatenate([xregion1, psi_region1])
          _ = layers.concatenate([xregion, psi_region])

        layer_ = layers.Dense(lsize, activation, name=name, **layer_kargs)
        # NOTE reordered from 1,T
        xregion = layer_(_)  #Z
        xregion1 = layer_(_1)  #Z
        if xdropout and l < num_layers-1:
          ldrop = layers.Dropout(dropout_rate)
          xregion1 = ldrop(xregion1)
          xregion = ldrop(xregion)
      x_next1.append(xregion1)
      x_next.append(xregion)

    if len(x_next) > 1:
      x_next1 = layers.Add()(x_next1)
      x_next = layers.Add()(x_next)
    else:
      x_next1 = x_next1[0]
      x_next = x_next[0]

    # Input component gets added in
    if input_layers is not None:
      if input_regions == 'all' or (isinstance(input_regions, list)
                                    and i in input_regions):
        # When input func type is "together" this is just [us]
        for j, u in enumerate(split_us):
          u1 = split_us1[j]
          b1 = u1
          b = u
          if input_type == "conditional":
            b1 = layers.concatenate([split_xs1[i], u1])
            b = layers.concatenate([split_xs[i], u])
          for l, (lsize, activation) in enumerate(input_layers):
            name = None
            if l == num_input_layers-1:
              name = "{}-u-{}".format(i, j)
            layer_ = layers.Dense(lsize, activation, name=name, use_bias=False,
                                  trainable=train_u, **layer_kargs)
            if zcond:
              b1 = layers.concatenate([b1, z1])
              b = layers.concatenate([b, z])

            b = layer_(b)  #Z
            b1 = layer_(b1)  #Z
            if not train_u:
              ws = layer_.get_weights()
              layer_.set_weights([np.zeros_like(_)+1.0 for _ in ws])
          x_next1 += b1
          x_next += b
    xs_next1.append(x_next1)
    xs_next.append(x_next)

  xs_next1 = layers.concatenate(xs_next1)
  xs_next = layers.concatenate(xs_next)
  model = tf.keras.models.Model(inputs=inputs, outputs=xs_next)
  model1 = tf.keras.models.Model(inputs=inputs1, outputs=xs_next1)
  return model1, model


def build_mr_dynamics_net(latent_region_sizes, seq_len, input_len,
                          dynamics_layers, input_layers=None, input_regions='all',
                          communication_type="additive",
                          input_type="additive", train_u=True,
                          input_func_type='seperate', # alt: together
                          kernel_initializer="glorot_uniform",
                          bias_initializer="random_uniform", zcond=False,
                          num_states=1, xdropout=False, dropout_rate=0.2, layer_reg=False,
                          seed=0, step_view=False, linear_communication=False):
  """Helper function for building an additive multi-region multi-layer network."""

  # Post iclr
  np.random.seed(seed)
  tf.random.set_seed(seed)

  layer_kargs = {"kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer}

  if layer_reg:
    layer_kargs['kernel_regularizer'] = regularizers.L2(1e-4)
    layer_kargs['bias_regularizer'] = regularizers.L2(1e-4)
    layer_kargs['activity_regularizer'] = regularizers.L2(1e-5)

  # -- Inputs --

  # Split latents by region
  num_regions = len(latent_region_sizes)
  if seq_len == 1:
    xs = tf.keras.Input(shape=(np.sum(latent_region_sizes),))
  else:
    xs = tf.keras.Input(shape=(seq_len, np.sum(latent_region_sizes),))
  split = lambda x: tf.split(x, num_or_size_splits=latent_region_sizes, axis=-1)
  split_xs = layers.Lambda(split)(xs)

  if input_layers is not None:
    if seq_len == 1:
      us = Input(shape=(input_len,))
    else:
      us = Input(shape=(seq_len, input_len,))
    inputs = [xs, us]
    split_us = [us]
    if input_func_type == "seperate":
      split = lambda u: tf.split(u, num_or_size_splits=input_len, axis=-1)
      split_us = layers.Lambda(split)(us)
  else:
    inputs = [xs]

  # NOTE could actually make this always conditioned, just pass in 0 if no zcond.
  if zcond:
    z = tf.keras.Input(shape=(seq_len, num_states,))
    inputs = [*inputs, z]

  if step_view:
    xs2 = tf.keras.Input(shape=(1, np.sum(latent_region_sizes),))
    split = lambda x: tf.split(x2, num_or_size_splits=latent_region_sizes, axis=-1)
    split_xs2 = layers.Lambda(split)(xs2)

    if input_layers is not None:
      us2 = Input(shape=(1, input_len,))
    inputs2 = [xs2, us2]
    split_us2 = [us2]
    if input_func_type == "seperate":
      split = lambda u: tf.split(us2, num_or_size_splits=input_len, axis=-1)
      split_us2 = layers.Lambda(split)(us2)
    else:
      inputs2 = [xs2]

  # -- Dynamics --

  num_layers = len(dynamics_layers)
  num_input_layers = len(input_layers)

  xs_next = []
  for i in range(num_regions):

    xlocal = split_xs[i]

    # Additive local dynamics and communication from each region
    x_next = []
    for j in range(num_regions):
      xregion = split_xs[j]
      for l, (lsize, activation) in enumerate(dynamics_layers):
        if l == num_layers-1: # final layer
          name = "{}-{}".format(i, j)
        else:
          name = None
        if j == i:
          if zcond:
            xregion = layers.concatenate([xregion, z])
          xregion = layers.Dense(lsize, activation, name=name, **layer_kargs)(xregion) #Z
        else:
          if communication_type == "additive-linear":
            print('linear comms', l)
            #if l < num_layers-1:  # for keeping only final layer
            #  continue
            activation = None
          _ = xregion
          if communication_type == "conditional":
            _ = layers.concatenate([xregion, xlocal])
          if zcond:
            _ = layers.concatenate([_, z])
          xregion = layers.Dense(lsize, activation, name=name, **layer_kargs)(_)  #Z
        if xdropout and l < num_layers-1:
          xregion = layers.Dropout(dropout_rate)(xregion)
      x_next.append(xregion)

    if len(x_next) > 1:
      x_next = layers.Add()(x_next)
    else:
      x_next = x_next[0]

    # Input component gets added in
    if input_layers is not None:
      if input_regions == 'all' or (isinstance(input_regions, list)
                                    and i in input_regions):
        # When input func type is "together" this is just [us]
        for j, u in enumerate(split_us):
          b = u
          if input_type == "conditional":
            b = layers.concatenate([split_xs[i], u])
          for l, (lsize, activation) in enumerate(input_layers):
            name = None
            if l == num_input_layers-1:
              name = "{}-u-{}".format(i, j)
            layer_ = layers.Dense(lsize, activation, name=name, use_bias=False,
                                  trainable=train_u, **layer_kargs)
            if zcond:
              b = layers.concatenate([b, z])
            b = layer_(b)  #Z
            if not train_u:
              ws = layer_.get_weights()
              layer_.set_weights([np.zeros_like(_)+1.0 for _ in ws])
          x_next += b
    xs_next.append(x_next)

  xs_next = layers.concatenate(xs_next)
  model = tf.keras.models.Model(inputs=inputs, outputs=xs_next)
  if step_view:
    model1 = tf.keras.models.Model(inputs=inputs, outputs=xs_next)
    return model, model1
  else:
    return model


def build_mr_emission_net(latent_region_sizes, seq_len, dense_layers, final_layers,
                          region_sizes, num_neurons,
                          kernel_initializer="glorot_uniform",
                          bias_initializer="random_uniform", layer_reg=False,
                          xdropout=False, dropout_rate=0.2): # NOTE not needed?, dropout_region=None):
  """Helper function for building a multi-region emission network."""

  layer_kargs = {"kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer}

  if layer_reg:
    layer_kargs['kernel_regularizer'] = regularizers.L2(1e-4)
    layer_kargs['bias_regularizer'] = regularizers.L2(1e-4)
    layer_kargs['activity_regularizer'] = regularizers.L2(1e-5)

  # Split by region
  num_regions = len(latent_region_sizes)
  num_observed_regions = len(region_sizes)  # If we want to add an extra latent region
  xs = Input(shape=(seq_len, np.sum(latent_region_sizes),))
  split = lambda x: tf.split(x, num_or_size_splits=latent_region_sizes, axis=-1)
  split_xs = layers.Lambda(split)(xs)

  # For each *observed* region, produce an emission.
  ys = []
  for j in range(num_observed_regions):
    xregion = split_xs[j]
    for lsize, activation in dense_layers:
        xregion = layers.Dense(lsize, activation, **layer_kargs)(xregion)
        if xdropout:
          xregion = layers.Dropout(dropout_rate)(xregion)
    # Num neurons in output layer depend on region
    lsize, activation = final_layers[j]
    yregion = layers.Dense(lsize, activation, **layer_kargs)(xregion)
    ys.append(yregion)
  ys = layers.concatenate(ys)

  # Pad emissions if fewer neurons than multiday max
  diff = num_neurons - np.sum(region_sizes)
  if diff > 0:
    # Transpose axes and back since zeropad func works on second to last dim
    ys = tf.transpose(ys, [0,2,1])
    ys = layers.ZeroPadding1D(padding=(0,diff))(ys)
    ys = tf.transpose(ys, [0,2,1])
  model = models.Model(inputs=[xs], outputs=ys)
  return model
