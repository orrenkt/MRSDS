
import yaml

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


def attr_dict_recursive(mydict):
  """Processing for nested yaml configs."""
  mydict = AttrDict(mydict)
  for k, v in mydict.items():
    if isinstance(v, dict):
      v = AttrDict(v)
      mydict[k] = attr_dict_recursive(v)
  return mydict


def load_yaml_config(config_path):
  with open(config_path, 'r') as stream:
    config = yaml.safe_load(stream)
    config = attr_dict_recursive(config)
  return config


class ConfigDict(dict):
  """Configuration dictionary that allows the `.` access."""

  def __init__(self, *args, **kwargs):
    super(ConfigDict, self).__init__(*args, **kwargs)
    for arg in args:
      if isinstance(arg, dict):
        for k, v in arg.iteritems():
          self[k] = v
    if kwargs:
      for k, v in kwargs.iteritems():
        self[k] = v

  def __getattr__(self, attr):
    return self.get(attr)

  def __setattr__(self, key, value):
    self.__setitem__(key, value)

  def __setitem__(self, key, value):
    super(ConfigDict, self).__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, item):
    self.__delitem__(item)

  def __delitem__(self, key):
    super(ConfigDict, self).__delitem__(key)
    del self.__dict__[key]


def get_data_config(batch_size):
  data_config = ConfigDict()
  data_config.batch_size = batch_size
  return data_config


def get_distribution_config(cov_mat=None, triangular_cov=False, trainable_cov=False,
                            raw_sigma_bias=0., sigma_min=1.e-5, sigma_scale=0.05):
  """Create default config for a multivariate normal gaussian distribution."""
  config = ConfigDict()
  config.cov_mat = cov_mat
  config.use_triangular_cov = triangular_cov
  config.use_trainable_cov = trainable_cov
  config.raw_sigma_bias = raw_sigma_bias
  config.sigma_min = sigma_min
  config.sigma_scale = sigma_scale
  return config


def get_learning_rate_config(flat_learning_rate=True, inverse_annealing_lr=False,
                             learning_rate=1.e-3, decay_alpha=1.e-2, decay_steps=20000,
                             warmup_steps=5000, warmup_start_lr=1.e-5):
  """Create default config for learning rate."""
  config = ConfigDict()
  config.flat_learning_rate = flat_learning_rate
  config.inverse_annealing_lr = inverse_annealing_lr
  config.learning_rate = learning_rate
  config.decay_alpha = decay_alpha
  config.decay_steps = decay_steps
  config.warmup_steps = warmup_steps
  config.warmup_start_lr = warmup_start_lr
  return config


def get_temperature_config(decay_steps=20000, decay_rate=0.99,
                           initial_temperature=1.e+3,
                           minimal_temperature=1.0, kickin_steps=10000,
                           use_temperature_annealing=True):
  """Create default config for temperature annealing."""
  config = ConfigDict()
  config.decay_steps = decay_steps
  config.decay_rate = decay_rate
  config.initial_temperature = initial_temperature
  config.minimal_temperature = minimal_temperature
  config.kickin_steps = kickin_steps
  config.use_temperature_annealing = use_temperature_annealing
  return config


def get_configs(cft):
  """Construct and return configs."""

  learning_rate_config = get_learning_rate_config(
    flat_learning_rate=cft.flat_learning_rate,
    inverse_annealing_lr=cft.use_inverse_annealing_lr,
    decay_steps=cft.num_steps,
    learning_rate=cft.learning_rate,
    warmup_steps=cft.lr_warmup_steps)

  learning_rate_config_do = None
  if hasattr(cft, 'dynamics_only'):
    learning_rate_config_do = get_learning_rate_config(
      flat_learning_rate=cft.flat_learning_rate,
      inverse_annealing_lr=cft.use_inverse_annealing_lr,
      decay_steps=cft.num_steps_do,
      learning_rate=cft.learning_rate,
      warmup_steps=cft.lr_warmup_steps_do)

  temperature_config = get_temperature_config(
    decay_rate=cft.annealing_rate,
    decay_steps=cft.annealing_steps,
    initial_temperature=cft.t_init,
    minimal_temperature=cft.t_min,
    kickin_steps=cft.annealing_kickin_steps,
    use_temperature_annealing=cft.temperature_annealing)

  train_data_config = get_data_config(batch_size=cft.batch_size)
  test_data_config = get_data_config(batch_size=cft.batch_size)

  return (learning_rate_config, learning_rate_config_do,
          temperature_config, train_data_config, test_data_config)
