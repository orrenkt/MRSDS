"""
Forward-backward algorithm for marginalizing over discrete states z
in mean field inference case.
"""

import tensorflow as tf
from mrsds.utils_tensor import (normalize_logprob, write_updates_to_tas,
                                clamp_probs, mask_logprob)


def forward_pass(log_a, log_b, logprob_z0, mask=None, debug=False):
  """
  Forward pass of Baum-Welch Algorithm (FB for HMMs).
  Computes the forward probability or filtering distribution p(z_t | x_{1:t}, y_{1:t})
  As well as the normalizer
  By employing log-exp-sum trick, values are computed in log space, including
  the output. Notation is adopted from https://arxiv.org/abs/1910.09588.
  Forward pass calculates the filtering likelihood of `log p(z_t | y_1:t)`.

  Args:
    log_a: float tensor, [batch_size, num_steps, num_states, num_states]
      Log likelihood of discrete transitions log p(z[t] | z[t-1], x[t-1])
      Note this is from all to all, eg log_a[:,t,i,j] is from z[t-1]=j to z[t]=i
    log_b: float tensor, [batch_size, num_steps, num_states]
      Log likelihood of emissions in the HMM log p(y[t], x[t] | z[t], x[t-1])
      Note this is for each state eg log_b[:,t,j] is for z[t]=j
    logprob_z0: float tensor [num_states],
      Initial discrete state probability log p(z[0])
    mask: optional, [batch_sizes, num_steps]. Mask timepoints for specific trials

  Returns:
    forward_pass: float tensor, [batch, num_steps, num_states]
      Filtering distribution log p(z_t | x_{1:t}, y_{1:t})
    normalizer: float tensor, [batch, num_steps]
      These are the alpha vars log p ( y_t, x_t | y_{1:t-1}, x_{1:t-1} )
      Summing over these gives alpha_T
  """
  num_steps = log_a.get_shape().with_rank_at_least(3).dims[1].value
  tas = [tf.TensorArray(tf.float32, num_steps, name=n)
         for n in ["forward_prob", "normalizer"]]

  # The function will return normalized forward probability and
  # normalizing constant as a list, [forward_logprob, normalizer].
  init_ = logprob_z0[:] + log_b[:,0,:]
  init_updates = normalize_logprob(init_, axis=-1)
  tas = write_updates_to_tas(tas, 0, init_updates)
  prev_prob = init_updates[0]
  init_state = (1, prev_prob, tas)

  def _cond(t, *unused_args):
    return t < num_steps

  def _steps(t, prev_prob, fwd_tas):
    bi_t = log_b[:, t, :]      # log p(y[t], x[t] | z[t], x[t-1])
    aij_t = log_a[:, t, :, :]  # log p(z[t] | x[t-1], z[t-1])
    _ = bi_t[:,:,tf.newaxis] + aij_t + prev_prob[:,tf.newaxis,:]
    current_updates = tf.math.reduce_logsumexp(_, axis=-1)
    if mask is not None:  # Comes before normalizer calc
      current_updates = mask_logprob(current_updates, mask[:,t,tf.newaxis], 'f')
    current_updates = normalize_logprob(current_updates, axis=-1)
    prev_prob = current_updates[0]
    fwd_tas = write_updates_to_tas(fwd_tas, t, current_updates)
    return (t+1, prev_prob, fwd_tas)

  _, _, tas_final = tf.while_loop(_cond, _steps, init_state)

  # Transpose to [batch, step, state], ([batch, step] for normalizer)
  forward_prob = tf.transpose(tas_final[0].stack(), [1,0,2])
  normalizer = tf.transpose(tf.squeeze(tas_final[1].stack(), axis=[-1]), [1,0])
  return forward_prob, normalizer


def backward_pass(log_a, log_b, mask=None, debug=False):
  """
  Backward pass of Baum-Welch Algorithm (FB for HMMs).
  Computes the backward probability log p(z_t | y_{t+1:T}, x_{t+1:T}), which
  will be combined with the forward probability to obtain the smoothing distribution.

  Args:
    log_a: float tensor, [batch_size, num_steps, num_states, num_states]
      Log likelihood of discrete transitions log p(z[t] | z[t-1], x[t-1])
      Note this is from all to all, eg log_a[:,t,i,j] is from z[t-1]=j to z[t]=i
    log_b: float tensor, [batch_size, num_steps, num_states]
      Log likelihood of emissions in the HMM log p(y[t], x[t] | z[t], x[t-1])
      Note this is for each state eg log_b[:,t,j] is for z[t]=j
    logprob_z0: float tensor [num_states],
      Initial discrete state probability log p(z[0])
    mask: optional, [batch_sizes, num_steps]. Mask timepoints for specific trials

  Returns:
    backward_pass: float tensor, [batch, num_steps, num_states]
      Backward pass distribution log p(z_t | x_{t+1:T}, y_{t+1:T})
  """
  batch_size, num_steps, num_states = tf.unstack(tf.shape(log_a)[:3])
  tas = [tf.TensorArray(tf.float32, num_steps, name=n)
         for n in ["backward_prob", "normalizer"]]
  init_updates = [tf.zeros([batch_size, num_states], dtype=tf.float32),
                  tf.zeros([batch_size, 1], dtype=tf.float32)]
  tas = write_updates_to_tas(tas, num_steps-1, init_updates)
  next_prob = init_updates[0]
  init_state = (num_steps-2, next_prob, tas)

  def _cond(t, *unused_args):
    return t > -1

  def _steps(t, next_prob, bwd_tas):
    """One step backward."""
    bi_tp1 = log_b[:, t+1, :]      # log p(y[t+1] | z[t+1])
    aij_tp1 = log_a[:, t+1, :, :]  # log p(z[t+1] | z[t], x[t])
    _ = next_prob[:,:,tf.newaxis] + aij_tp1 + bi_tp1[:,:,tf.newaxis]
    current_updates = tf.math.reduce_logsumexp(_, axis=-2)
    if mask is not None:  # Comes before normalizer calc
      current_updates = mask_logprob(current_updates, mask[:,t,tf.newaxis], 'b')
    current_updates = normalize_logprob(current_updates, axis=-1)
    next_prob = current_updates[0]
    bwd_tas = write_updates_to_tas(bwd_tas, t, current_updates)
    return (t-1, next_prob, bwd_tas)

  _, _, tas_final = tf.while_loop(_cond, _steps, init_state)

  backward_prob = tf.transpose(tas_final[0].stack(), [1,0,2])
  return backward_prob


def compute_posterior(fwd, bwd, log_a, log_b, temp=1.0):
  """
  `posterior = p(z[t]=k | z[1:T], x[1:T])'
  `log_gamma2 = p(z[t]=k, z[t-1]=j | z[1:T], x[1:T])'
  """
  batch_size, maxlen = tf.unstack(tf.shape(log_a)[:2])
  num_states = tf.shape(log_a)[-1]

  m_fwd = fwd[:,:-1,tf.newaxis,:]
  m_bwd = bwd[:,1:,:,tf.newaxis]
  m_a = log_a[:,1:,:,:]
  m_b = log_b[:,1:,:,tf.newaxis]

  # Compute posterior single and pair state probabilities
  posterior = normalize_logprob(fwd + bwd, axis=-1, temperature=temp)[0]
  gamma_ij = normalize_logprob(m_fwd + m_a + m_bwd + m_b, axis=[-2,-1])[0]

  # Padding the first timepoint
  pad = tf.zeros([batch_size, 1, num_states, num_states], dtype=tf.float32)
  gamma_ij = tf.concat([pad, gamma_ij], axis=1)

  return posterior, gamma_ij


def forward_backward(log_a, log_b, log_init, mask=None, temp=1.0):
  """Forward backward algorithm."""

  # Run forward backwards
  inputs = (log_a, log_b, log_init, mask)
  fwd, fwd_normalizer = forward_pass(*inputs)
  bwd = backward_pass(*inputs[:2], mask)

  if mask is not None:
    fwd = mask_logprob(fwd, mask[:,:,tf.newaxis], 'fwd')
    fwd_normalizer = mask_logprob(fwd_normalizer, mask, 'fwd norm')
    bwd = mask_logprob(bwd, mask[:,:,tf.newaxis], 'bwd')

  # Compute posteriors
  inputs = (fwd, bwd, log_a, log_b)
  posterior, gamma_ij = compute_posterior(*inputs, temp)

  if mask is not None:
    posterior = mask_logprob(posterior, mask[:,:,tf.newaxis], 'pos')

  log_p_xy = tf.reduce_sum(fwd_normalizer, axis=-1)
  return fwd, bwd, posterior, gamma_ij, log_p_xy
