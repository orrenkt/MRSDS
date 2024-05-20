"""."""

import numpy as np
import tensorflow as tf


@tf.autograph.experimental.do_not_convert  # TF yells otherwise..
def tensor_for_ta(input_ta, swap_batch_time=True):
  """Creates a `Tensor` for the input `TensorArray`."""
  if swap_batch_time:
    res = input_ta.stack()
    _ = np.arange(2, res.shape.ndims)
    return tf.transpose(res, np.concatenate([[1, 0], _]))
  else:
    return input_ta.stack()


@tf.autograph.experimental.do_not_convert  # TF yells otherwise..
def write_updates_to_tas(tensor_arrays, t, tensor_updates):
  """Write updates to corresponding TensorArrays at time step t."""
  assert len(tensor_arrays) == len(tensor_updates)
  num_updates = len(tensor_updates)

  return [tensor_arrays[i].write(t, tensor_updates[i]) #.mark_used()
          for i in range(num_updates)]


@tf.autograph.experimental.do_not_convert  # TF yells otherwise..
def clamp_probs(tensor):
  eps = 1e-9  #get_precision(tensor)
  return tf.clip_by_value(tensor, eps, 1-eps)


@tf.autograph.experimental.do_not_convert  # TF yells otherwise..
def normalize_logprob(logmat, axis=-1, temperature=1.0):
  """Normalizing log probability with `reduce_logsumexp`."""
  logmat = tf.convert_to_tensor(logmat, dtype_hint=tf.float32)
  logmat = logmat / temperature
  normalizer = tf.math.reduce_logsumexp(logmat, axis=axis, keepdims=True)
  return [logmat-normalizer, normalizer]


def mask_logprob(tensor, mask, name):
  masked = tensor * mask
  return masked
