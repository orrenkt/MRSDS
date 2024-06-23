
from itertools import product

import matplotlib
matplotlib.use('Agg')  # Save plots to file
import matplotlib.pyplot as plt
import scipy
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from mrsds.utils_analysis import (get_communication_models, get_msgs_and_norms,
                                  generate_trajectories, fit_xs,
                                  per_timepoint_r2, transform_latents, r2_score,
                                  inv_transform_latents, generate_1step_trajectories,
                                  eval_cosmooth_mse, eval_cosmooth_r2)


def learning_rate_warmup(global_step, warmup_end_lr,
                         warmup_start_lr, warmup_steps,
                         dtype=tf.float32):
  """Linear learning rate warm-up."""
  p = tf.cast(global_step, dtype) / tf.cast(warmup_steps, dtype)
  diff = warmup_end_lr - warmup_start_lr
  return warmup_start_lr + diff*p


def learning_rate_schedule(global_step, config):
  """Learning rate schedule with linear warm-up and cosine decay."""
  warmup_schedule = learning_rate_warmup(global_step=global_step,
                                         warmup_end_lr=config.learning_rate,
                                         warmup_start_lr=config.warmup_start_lr,
                                         warmup_steps=config.warmup_steps)
  diff = config.decay_steps-config.warmup_steps
  f = tf.keras.experimental.CosineDecay(initial_learning_rate=config.learning_rate,
                                        decay_steps=diff,
                                        alpha=config.decay_alpha, name=None)
  decay_schedule = f(tf.math.maximum(global_step - config.warmup_steps, 0))
  return tf.cond(global_step < config.warmup_steps,
                 lambda: warmup_schedule,
                 lambda: decay_schedule)


def get_learning_rate(learning_rate_config, global_step):
  """Construct Learning Rate Schedule."""
  return learning_rate_schedule(global_step, learning_rate_config)


def schedule_exponential_decay(global_step, config, min_val=1e-10, dtype=tf.float32):
  """Flat and exponential decay schedule."""
  global_step = tf.cast(global_step, dtype)
  decay_steps = tf.cast(config.decay_steps, dtype)
  kickin_steps = tf.cast(config.kickin_steps, dtype)
  _ = tf.math.maximum(global_step - kickin_steps, 0.0) / decay_steps
  decay_schedule = config.initial_temperature**2 * config.decay_rate**_
  temp_schedule = tf.cond(global_step < config.kickin_steps,
                          lambda: config.initial_temperature,
                          lambda: tf.maximum(decay_schedule, min_val))
  return temp_schedule


def get_temperature(temperature_config, step):
  """Construct Temperature Annealing Schedule."""
  if temperature_config.use_temperature_annealing:
    temperature_schedule = schedule_exponential_decay(
      step, temperature_config, temperature_config.minimal_temperature)
  else:
    temperature_schedule = temperature_config.initial_temperature
  return temperature_schedule


def eval_perf(mrsds_model, train_final_batch, test_batch, cosmooth_batch,
              trial_lengths_train, trial_lengths_test, dropout_idxs):
  """
  Save metrics mse and r2 on reconstruction, cosmooth. Also 1 step gen.
  This is only for train final batch, test batch, cosmooth batch.
  """
  names = ['ys', 'us', 'masks']
  batch_dict_train = dict(zip(names, train_final_batch[:3]))
  batch_dict_test = dict(zip(names, test_batch[:3]))
  batch_dict_cosmooth = dict(zip(names, cosmooth_batch[:3]))

  # Run inference
  train_final_result = mrsds_model(**batch_dict_train)
  test_result = mrsds_model(**batch_dict_test)
  cosmooth_result = mrsds_model(**batch_dict_cosmooth)
  tfr = train_final_result
  tr = test_result
  cr = cosmooth_result

  # --- Compute reconstruction accuracy metrics ---

  # Final train batch
  train_lengths = trial_lengths_train[:len(trial_lengths_test)]
  inputs = [tfr['ys'].numpy(), tfr['reconstructed_ys'].numpy(),
            train_lengths]
  mse_train_final = eval_cosmooth_mse(*inputs)
  r2_train_final = eval_cosmooth_r2(*inputs)

  # Test
  inputs = [tr['ys'].numpy(), tr['reconstructed_ys'].numpy(),
            trial_lengths_test]
  mse_test = eval_cosmooth_mse(*inputs)
  r2_test = eval_cosmooth_r2(*inputs)

  # Cosmooth
  inputs = [tr['ys'].numpy(), cr['reconstructed_ys'].numpy(),
            trial_lengths_test]
  mse_cosmooth = eval_cosmooth_mse(*inputs, dropout_idxs)
  r2_cosmooth = eval_cosmooth_r2(*inputs, dropout_idxs)

  # --- Generate trajectories and compute gen accuracy metrics ---

  # Check if zcond, psis
  zcond = False
  psis = False
  dynamics_type = str(type(mrsds_model.x_tran))
  if 'Zcond' in dynamics_type:
    zcond = True
  if 'Psis' in dynamics_type:
    psis = True
  num_states = mrsds_model.num_states

  # Train final batch
  inputs = [tfr['x_sampled'], tfr['us']]
  args = {}
  if num_states > 1:
    args['zs_post'] = tfr['z_posterior_ll']
  if psis:
    args['psis'] = tfr['psi_sampled']
  xs_gen1_train, ys_gen1_train_final = generate_1step_trajectories(mrsds_model, *inputs, **args)
  inputs = [tfr['ys'].numpy(), ys_gen1_train_final, train_lengths]
  mse_gen1_train_final = eval_cosmooth_mse(*inputs)
  r2_gen1_train_final = eval_cosmooth_r2(*inputs)

  # Test
  inputs = [tr['x_sampled'], tr['us']]
  args = {}
  if num_states > 1:
    args['zs_post'] = tr['z_posterior_ll']
  if psis:
    args['psis'] = tr['psi_sampled']
  xs_gen1_test, ys_gen1_test = generate_1step_trajectories(mrsds_model, *inputs, **args)
  inputs = [tr['ys'].numpy(), ys_gen1_test, trial_lengths_test]
  mse_gen1_test = eval_cosmooth_mse(*inputs)
  r2_gen1_test = eval_cosmooth_r2(*inputs)

  # Cosmooth
  inputs = [cr['x_sampled'], cr['us']]
  args = {}
  if num_states > 1:
    args['zs_post'] = cr['z_posterior_ll']
  if psis:
    args['psis'] = cr['psi_sampled']
  xs_gen1_cosmooth, ys_gen1_cosmooth = generate_1step_trajectories(mrsds_model, *inputs, **args)
  inputs = [cr['ys'].numpy(), ys_gen1_cosmooth, trial_lengths_test]
  mse_gen1_cosmooth = eval_cosmooth_mse(*inputs, dropout_idxs)
  r2_gen1_cosmooth = eval_cosmooth_r2(*inputs, dropout_idxs)

  return (train_final_result, test_result, cosmooth_result,
          mse_train_final, mse_test, mse_cosmooth,
          r2_train_final, r2_test, r2_cosmooth,
          mse_gen1_train_final, mse_gen1_test, mse_gen1_cosmooth,
          r2_gen1_train_final, r2_gen1_test, r2_gen1_cosmooth)


def save_results2(fpath, mrsds_model, x_transition_networks, ys_train,
                  us_train, masks_train, ys_test, us_test, latent_region_sizes,
                  trial_lengths_train, trial_lengths_test, train_final_batch,
                  test_batch, cosmooth_batch, dropout_idxs, train_idxs,
                  test_idxs, cxt, tracking):
  """
  -Save all latents (x, z) and reconstructions for the entire dataset (train+test).
  -TODO:Save messages decomposition for the entire dataset
  -save train tracking variables dict.
  """
  # NOTE should move one level up and pass in comm models not xtran nets.
  num_regions = len(latent_region_sizes)
  xdim = latent_region_sizes[0]
  input_len = us_train[0].shape[-1]

  # --- Train ---

  # NOTE: need to double check this, should only mask padded neurons and timepoints
  masks_train_int = [_.astype(int) for _ in masks_train]

  all_xs = []
  all_qzs = []
  all_zs = []
  all_msgs = []
  all_norms = []
  all_ys_recon = []
  all_logas = []
  all_logbs = []
  #all_lls = []

  _ = np.arange(0, len(ys_train), 32)
  starts_ends2 = list(zip(_, list(_[1:]) + [len(ys_train)]))
  print('ses', starts_ends2)
  for start, end in starts_ends2:

    nt = end-start
    tensor_inputs = [tf.convert_to_tensor(ys_train[start:end], tf.float32),
                     tf.convert_to_tensor(us_train[start:end], tf.float32),
                     tf.convert_to_tensor(masks_train_int[start:end],tf.float32)]
    names = ['ys', 'us', 'masks']
    result = mrsds_model(**dict(zip(names,tensor_inputs)))
    msgs = tf.zeros(1)
    norms = tf.zeros(1)
    all_ys_recon.extend(tf.unstack(tf.squeeze(result['reconstructed_ys']).numpy()))
    all_xs.extend(tf.unstack(tf.squeeze(result['x_sampled']).numpy()))
    all_qzs.extend(np.exp(tf.unstack(result['z_posterior_ll'].numpy())))
    all_msgs.extend(msgs)
    all_norms.extend(norms)
    all_logas.extend(tf.unstack(result['log_a'].numpy()))
    all_logbs.extend(tf.unstack(result['log_b'].numpy()))
    if 'zs_logprob' in result:
      all_zs.extend(tf.unstack(result['zs_logprob'].numpy()))

  # --- Test ---

  batch_dict = dict(zip(names, test_batch[:3]))
  test_result = mrsds_model(**batch_dict)
  msgs = tf.zeros(1)
  norms = tf.zeros(1)
  ys_recon_test = tf.unstack(tf.squeeze(test_result['reconstructed_ys']).numpy())
  xs_sampled_test = tf.unstack(tf.squeeze(test_result['x_sampled']).numpy())
  qzs_test = np.exp(tf.unstack(test_result['z_posterior_ll'].numpy()))
  msgs_test = msgs
  norms_test = norms
  loga_test = tf.unstack(test_result['log_a'].numpy())
  logb_test = tf.unstack(test_result['log_b'].numpy())
  zs_test_ = 0
  if 'zs_logprob' in result:
    zs_test_ = tf.unstack(result['zs_logprob'].numpy())

  mdict = {
    'xs_train': all_xs,
    'xs_test': xs_sampled_test,
    'zs_train': all_qzs,
    'zs_test': qzs_test,
    'zs_logprob': all_zs,
    'zs_logprob_test': zs_test_,
    'us_train': us_train,
    'us_test': us_test,
    'ys_train': ys_train,
    'ys_test': ys_test,
    'ys_recon_train': all_ys_recon,
    'ys_recon_test': ys_recon_test,
    'msgs_train': all_msgs,
    'msgs_test': msgs_test,
    'norms_train': all_norms,
    'norms_test': norms_test,
    'train_ids': train_idxs,
    'test_ids': test_idxs,
    'train_lengths': trial_lengths_train,
    'test_lengths': trial_lengths_test,
    'latent_region_sizes': latent_region_sizes,
    'loga_train': all_logas,
    'logb_train': all_logbs,
    'loga_test': loga_test,
    'logb_test': logb_test,
  }
  mdict.update(tracking)

  scipy.io.savemat(fpath, mdict)
  print('Saved latents etc.')
  return


def save_results(fpath, mrsds_model, x_transition_networks, ys_train,
                 us_train, masks_train, ys_history_train, ys_test, us_test,
                 true_latents, num_samples, num_states, latent_region_sizes,
                 trial_lengths_train, trial_lengths_test, train_final_batch,
                 test_batch, cosmooth_batch, dropout_idxs, train_idxs,
                 test_idxs, trial_metadata, cxt, tracking, xs_train=None,
                 xs_test=None, zs_train=None, zs_test=None, get_msgs=True,
                 track_gen=True): #True): #False):
  """
  -Save all latents (x, z) for the entire dataset (train+test).
  -Save messages decomposition for the entire dataset
  -Save reconstructions for entire dataset.
  #### Save likelihoods for x, z, entropy anything else?
  #### Save flow fields.
  """
  # NOTE should move one level up and pass in comm models not xtran nets.
  num_regions = len(latent_region_sizes)
  xdim = latent_region_sizes[0]
  input_len = us_train[0].shape[-1]
  if mrsds_model.stages:
    input_len -= 1
  if get_msgs:
    communication_models = get_communication_models(x_transition_networks,
                                                    num_regions, num_states,
                                                    input_len)
  cond_dict = {}
  if cxt.dynamics.communication_type == "conditional":
    cond_dict["conditional_msgs"] = True
  if cxt.dynamics.input_type == "conditional":
    cond_dict["conditional_inputs"] = True
  #cond_dict["x0_model"] = x0_model


  # Run inference on all training

  # --- Train ---

  # NOTE: need to double check this, should only mask padded neurons and timepoints
  masks_train_int = [_.astype(int) for _ in masks_train]

  all_xs = []
  all_qzs = []
  all_msgs = []
  all_norms = []
  all_ys_recon = []
  all_logas = []
  all_logbs = []
  #all_lls = []

  _ = np.arange(0, len(ys_train), 32)
  starts_ends2 = list(zip(_, list(_[1:]) + [len(ys_train)]))
  for start, end in starts_ends2:

    nt = end-start
    hist = None
    if ys_history_train is not None:
      hist = tf.convert_to_tensor(ys_history_train[start:end], tf.float32)
    tensor_inputs = [tf.convert_to_tensor(ys_train[start:end], tf.float32),
                     tf.convert_to_tensor(us_train[start:end], tf.float32),
                     tf.convert_to_tensor(masks_train_int[start:end],tf.float32),
                     np.zeros(nt).astype(int),
                     np.zeros(nt).astype(int),
                     hist]
    if xs_train is not None:
      xs_ = np.array(xs_train[start:end])
      zs_ = np.array(zs_train[start:end])
      ys, batch_dict = make_batch_dict(tensor_inputs, xs_true=xs_, zs_true=zs_)
    else:
      ys, batch_dict = make_batch_dict(tensor_inputs)

    result = mrsds_model(ys, **batch_dict, temperature=1,
                         num_samples=num_samples)
    if get_msgs:
      msgs, norms = get_msgs_and_norms(communication_models, result, num_states,
                                       num_regions, xdim, latent_region_sizes,
                                       trial_lengths_train[start:end], input_len,
                                       **cond_dict)
    else:
      msgs = tf.zeros(1)
      norms = tf.zeros(1)
    all_ys_recon.extend(tf.unstack(tf.squeeze(result['reconstructed_ys']).numpy()))
    all_xs.extend(tf.unstack(tf.squeeze(result['x_sampled']).numpy()))
    all_qzs.extend(np.exp(tf.unstack(result['z_posterior_ll'].numpy())))
    all_msgs.extend(msgs)
    all_norms.extend(norms)
    all_logas.extend(tf.unstack(result['log_a'].numpy()))
    all_logbs.extend(tf.unstack(result['log_b'].numpy()))

  # --- Test ---

  ys, batch_dict = make_batch_dict(test_batch, xs_true=np.array(xs_test),
                                   zs_true=np.array(zs_test))
  test_result = mrsds_model(ys, **batch_dict, temperature=1,
                            num_samples=num_samples)
  if get_msgs:
    msgs, norms = get_msgs_and_norms(communication_models, test_result, num_states,
                                     num_regions, xdim, latent_region_sizes,
                                     trial_lengths_test, input_len, **cond_dict)
  else:
    msgs = tf.zeros(1)
    norms = tf.zeros(1)
  ys_recon_test = tf.unstack(tf.squeeze(test_result['reconstructed_ys']).numpy())
  xs_sampled_test = tf.unstack(tf.squeeze(test_result['x_sampled']).numpy())
  qzs_test = np.exp(tf.unstack(test_result['z_posterior_ll'].numpy()))
  msgs_test = msgs
  norms_test = norms
  loga_test = tf.unstack(test_result['log_a'].numpy())
  logb_test = tf.unstack(test_result['log_b'].numpy())

  tower_sums_train = np.sum(np.array(us_train), axis=1)
  train_left_idxs = np.where(tower_sums_train[:,0] > tower_sums_train[:,1])[0]
  train_right_idxs = np.where(tower_sums_train[:,0] < tower_sums_train[:,1])[0]
  train_same_idxs = np.where(tower_sums_train[:,0] == tower_sums_train[:,1])[0]

  tower_sums_test = np.sum(np.array(us_test), axis=1)
  test_left_idxs = np.where(tower_sums_test[:,0] > tower_sums_test[:,1])[0]
  test_right_idxs = np.where(tower_sums_test[:,0] < tower_sums_test[:,1])[0]
  test_same_idxs = np.where(tower_sums_test[:,0] == tower_sums_test[:,1])[0]

  if trial_metadata is not None:
    names = ['correct_ids', 'error_ids', 'towers_ids', 'left_ids',
             'right_ids', 'stages', 'excluded_ids']
    trial_metadata = dict(zip(names, trial_metadata))
  else:
    trial_metadata = {}

  extra = {}
  if xs_train is not None:
    extra = {'xs_train': np.array(xs_train), 'xs_test': np.array(xs_test)}
  perf = eval_perf(mrsds_model, train_final_batch, test_batch, cosmooth_batch,
                   trial_lengths_train, trial_lengths_test, dropout_idxs,
                   num_samples=num_samples, **extra, gen_1step=False)

  (_, _, _, train_mses, test_mses, cosmooth_mses,
   train_gen1_mses, test_gen1_mses, cosmooth_gen1_mses,
   train_r2s, test_r2s, cosmooth_r2s,
   train_gen1_r2s, test_gen1_r2s, cosmooth_gen1_r2s,
   train_lls, test_lls, cosmooth_lls) = perf #occupancies

  # Generate trajectories from dynamics and true IC.
  gen_dict = {}
  if track_gen and xs_train is not None:
    use_zs_true = False #True

    xs_train = np.array(xs_train)
    xs_test = np.array(xs_test)
    zs_train = np.array(zs_train)
    zs_test = np.array(zs_test)
    us_train = np.array(us_train)
    us_test = np.array(us_test)
    ys_train = np.array(ys_train)
    ys_test = np.array(ys_test)

    nt_train, tl, xdim = xs_train.shape
    ydim = ys.shape[-1]
    nt_test = len(xs_test)

    # Fit LRs
    lrs_train = []
    for i in range(1):
        s = (i*xdim)
        lr_train, _ = fit_xs(xs_train[:,:,s:s+xdim],
                             np.array(all_xs)[:,:,s:s+xdim])
        lrs_train.append(lr_train)

    # Generate trajectories from ICs.
    ics = xs_train[:,0,:]
    ics_trans = inv_transform_latents(lrs_train, ics, xdim)
    us_ = us_train
    zics = zs_train[:,0]
    zargs = {}
    if use_zs_true: # NOTE TODO
        #zargs['zs_true'] = np.argmax(dat['zs_train'], axis=-1)
        zargs['zs_true'] = zs_true
    ret = generate_trajectories(mrsds_model, xics=ics_trans, zics=zics,
                                T=tl, us=us_train, latent_dims=xdim,
                                obs_dims=ydim, zcond=True, **zargs)
    zs_gen, xs_gen, ys_gen = ret

    ics_test = xs_test[:,0,:]
    ics_test_trans = inv_transform_latents(lrs_train, ics_test)
    us_ = us_test #dats[i]
    zics_test = zs_test[:,0]
    zargs = {}
    if use_zs_true:
        #zargs['zs_true'] = np.argmax(dat['zs_test'], axis=-1)
        zargs['zs_true'] = zs_test
    ret = generate_trajectories(mrsds_model, xics=ics_test_trans, zics=zics_test,
                                T=tl, us=us_test, latent_dims=xdim,
                                obs_dims=ydim, zcond=True, **zargs)
    zs_gen_test, xs_gen_test, ys_gen_test = ret

    # Transform latents
    xs_train_inf_trans = transform_latents(lrs_train, np.array(all_xs))
    xs_test_inf_trans = transform_latents(lrs_train, np.array(xs_sampled_test))
    xs_train_gen_trans = transform_latents(lrs_train, xs_gen)
    xs_test_gen_trans = transform_latents(lrs_train, xs_gen_test)

    # Compute R^2 for both inf and gen on test
    r2s_gen_train = per_timepoint_r2(xs_train, xs_train_gen_trans)
    r2s_inf_train = per_timepoint_r2(xs_train, xs_train_inf_trans)
    r2s_gen_test = per_timepoint_r2(xs_test, xs_test_gen_trans)
    r2s_inf_test = per_timepoint_r2(xs_test, xs_test_inf_trans)

    r2sy_gen_train = per_timepoint_r2(ys_train, ys_gen)
    r2sy_gen_test = per_timepoint_r2(ys_test, ys_gen_test)
    r2sy_inf_train = per_timepoint_r2(ys_train, np.array(all_ys_recon))
    r2sy_inf_test = per_timepoint_r2(ys_test, np.array(ys_recon_test))

    gen_dict = {
      'r2s_gen_train': r2s_gen_train,
      'r2s_gen_test': r2s_gen_test,
      'r2s_inf_train': r2s_inf_train,
      'r2s_inf_test': r2s_inf_test,
      'r2sy_gen_train': r2sy_gen_train,
      'r2sy_gen_test': r2sy_gen_test,
      'r2sy_inf_train': r2sy_inf_train,
      'r2sy_inf_test': r2sy_inf_test,
      'zs_gen': zs_gen,
      'xs_gen_trans': xs_gen,
      'ys_gen': ys_gen,
      'zs_gen_test': zs_gen_test,
      'xs_gen_test_trans': xs_gen_test,
      'ys_gen_test': ys_gen_test
    }

  mdict = {
    **dict([(k+'_true',v) for k, v in true_latents.items()]),
    'xs_train': all_xs,
    'xs_test': xs_sampled_test,
    'zs_train': all_qzs,
    'zs_test': qzs_test,
    'us_train': us_train,
    'us_test': us_test,
    'ys_train': ys_train,
    'ys_test': ys_test,
    'ys_recon_train': all_ys_recon,
    'ys_recon_test': ys_recon_test,
    'msgs_train': all_msgs,
    'msgs_test': msgs_test,
    'norms_train': all_norms,
    'norms_test': norms_test,
    'train_ids': train_idxs,
    'test_ids': test_idxs,
    'train_lengths': trial_lengths_train,
    'test_lengths': trial_lengths_test,
    'towers_sum_train': tower_sums_train,
    'tower_sums_test': tower_sums_test,
    'train_left_ids': train_left_idxs,
    'train_right_ids': train_right_idxs,
    'test_left_ids': test_left_idxs,
    'test_right_ids': test_right_idxs,
    'train_same_ids': train_same_idxs,
    'test_same_ids': test_same_idxs,
    'latent_region_sizes': latent_region_sizes,
    'elbos': tracking['elbos'],
    #'lrs': np.array(tracking['lrs']),
    'loga_train': all_logas,
    'logb_train': all_logbs,
    'loga_test': loga_test,
    'logb_test': logb_test,
    #**trial_metadata
  }

  mdict.update(tracking)
  mdict.update(gen_dict)

  scipy.io.savemat(fpath, mdict)
  print('Saved latents etc.')
  return


@tf.function #(experimental_relax_shapes=True)
def train_step_modular_svae(batch_dict, mrsds_model, optimizer,
                            num_samples, num_days, objective, learning_rate,
                            temperature=tf.constant(1.0), smooth_penalty=False, smooth_coef=1,
                            dynamics_only=False, track_grads=False,
                            modular_lrs=[4,1,1,2,1], random_seed=131, #tf.constant(0),
                            beta=tf.constant(1)):
  """Runs one training step and returns metrics evaluated on the train set.

  Args:
    train_batch:  a batch of the training set.
    mrsds_model: tf.keras.Model, model to be trained.
    optimizer: tf.keras.optimizers.Optimizer,
      optimizer to use for back-propagation.
    num_samples: int, number of samples per trajectories to use at train time.
    objective: str, which objective to use ("elbo" or "iwae").
    learning_rate: float, learning rate to use for back-propagation.
    temperature: float, annealing temperature to use on the model.
    cross_entropy_coef: float, weight of the cross-entropy loss.

  Returns:
    Dictionary of metrics, {str: tf.Tensor}
      log_likelihood: value of the estimated train loglikelihood.
      cross_entropy: value of the estimated cross-entropy loss.
      objective: value of the estimate objective to minimize.
  """
  # NOTE: persistent just for debugging sub-grad magnitudes
  pgt = True if track_grads else False
  with tf.GradientTape(persistent=pgt) as tape:
    train_result = mrsds_model(**batch_dict, temperature=temperature,
                               random_seed=random_seed, do_batch=dynamics_only,
                               beta=beta)
    train_loglikelihood = train_result[objective]
    train_objective = -1. * (train_loglikelihood) #+ cross_entropy_coef * train_zxent)

    # NOTE problem when combined with new objective since py|x also changes.
    #if dynamics_only:
    #  train_objective = -1. * train_result['logpx_sum']  #logpxsum

    if smooth_penalty:
      train_objective += train_result['diffs_norm'] * smooth_coef

  train_vars = [_ for _ in mrsds_model.trainable_variables]
  if dynamics_only:
    xvars = mrsds_model.x_tran.trainable_variables
    if hasattr(mrsds_model, "z_tran"):
      zvars = mrsds_model.z_tran.trainable_variables
      xz0vars = list(mrsds_model.x0_dist.trainable_variables) #+ [mrsds_model.log_init_z]
    else:
      zvars = []
      xz0vars = list(mrsds_model.x0_dist.trainable_variables)
    yvars = mrsds_model.y_emit.trainable_variables
    qvars = mrsds_model.inference_net.trainable_variables
    train_vars = xvars + zvars + xz0vars + yvars

  if not pgt:
    # Apply grads
    grads_ = tape.gradient(train_objective, train_vars)
    grads = []
    for grad, var in zip(grads_, train_vars):
      if grad is None:
        grads.append(tf.zeros_like(var))
      else:
        grads.append(grad)
    clipped_grads = [tf.clip_by_value(grad, -10.0, 10.0) for grad in grads]
    optimizer.learning_rate = learning_rate
    optimizer.apply_gradients(list(zip(clipped_grads, train_vars)))

  else:

    # clear vars
    ys_train = []
    batch_dict = {}

    gradclip = lambda grads: [tf.where(tf.math.is_nan(grad),
                                       tf.zeros_like(grad),
                                       tf.clip_by_value(grad, -10.0, 10.0))
                              for grad in grads if grad is not None]
    normsum = lambda grads: tf.reduce_sum([tf.norm(grad) for grad in grads])
    normmean = lambda grads: tf.reduce_mean([tf.norm(grad) for grad in grads])

    # Clip grads.
    grads_xz = gradclip(tape.gradient(train_objective, xvars + zvars)) #+zvars
    grads_y = gradclip(tape.gradient(train_objective, yvars))
    grads_q = gradclip(tape.gradient(train_objective, qvars))
    grads_xz0 = gradclip(tape.gradient(train_objective, xz0vars))

    # Scale grads
    grads_xz = [grad * modular_lrs[0] for grad in grads_xz]
    grads_y = [grad * modular_lrs[1] for grad in grads_y]
    grads_q = [grad * modular_lrs[2] for grad in grads_q]
    grads_xz0 = [grad * modular_lrs[0] for grad in grads_xz0]

    if dynamics_only: # added y here for new objective, so only dropping encoder grads and vars.
      train_vars = xvars + zvars + xz0vars + yvars
      train_grads = grads_xz + grads_xz0 + grads_y
    else:
      train_vars = xvars + zvars + yvars + xz0vars + qvars
      train_grads = grads_xz + grads_y + grads_xz0 + grads_q

    # Apply grads
    optimizer.learning_rate = learning_rate
    optimizer.apply_gradients(list(zip(train_grads, train_vars)))

  train_result["objective"] = train_objective

  del tape
  return train_result
