
import os
import time
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import imp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_PROFILER'] = '1'
import tensorflow as tf
tf.autograph.set_verbosity(0)

import numpy as np
import tensorflow as tf

# NOTE should set this up properally
mrsds = imp.load_source('mrsds', './__init__.py')
from mrsds.utils_config import load_yaml_config, get_configs, get_learning_rate_config
from mrsds.utils_data import load_prep_data, construct_tf_datasets
from mrsds.utils_training import get_learning_rate, get_temperature, eval_perf, save_results2
from mrsds.utils_training import train_step_modular_svae as train_step


def run_training(data_path, result_path, config_path, name, gpu_id, num_states,
                 num_dims, mtype='mrsds', data_seed=0, model_seed=0,
                 training_seed=0):
  """Training function, to be called from command line. See example call script."""

  # MRSDS is mean field inference model (single or switching dynamics)
  # SVAE is structured inference model (single dynamics)
  # S-SVAE is structured inference model with switching dynamics
  if mtype == 'mrsds':
    from mrsds import mrsds as mrsds
  elif mtype == 'svae':
    from mrsds import mrsds_svae as mrsds
  elif mtype == 's-svae':
    from mrsds import mrsds_switching_svae as mrsds
  else:
    raise ValueError('Mtype not supported', mtype)
  svae = False
  if 'svae' in name:
    svae = True

  # Set GPU
  os.environ['tf_data_private_threadpool.thread_limit'] = '100'
  tf.config.run_functions_eagerly(True)
  gpus = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

  # NOTE single GPU by default. If model needs extra GPUs can manually set here
  # Should change command line arg parsing to support multiple gpu ids.
  #tf.config.experimental.set_visible_devices([gpus[gpu_id], gpus[7]], 'GPU')
  #tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
  #tf.config.experimental.set_memory_growth(gpus[7], True)

  config = load_yaml_config(config_path)
  cfd = config.data
  cft = config.training
  cfi = config.inference
  cfm = config.model
  cxt = cfm.xtransition

  log_dir = result_path + name + '-logs/'
  model_dir = result_path + name + '-models/'

  # -----  Load and prep data -----

  tf.random.set_seed(0)
  np.random.seed(0)

  # Load dataset
  ret = load_prep_data(cfd, cft, data_path, data_seed)
  (num_regions, region_sizes, ys_train, us_train,
   ys_test, us_test, ys_test_cosmooth, dropout_idxs,
   trial_lengths_train, trial_lengths_test, train_idxs, test_idxs,
   masks_train, masks_test, xs_train, xs_test, zs_train, zs_test) = ret

  trial_length = ys_train[0].shape[0]
  num_inputs = us_train[0].shape[1]

  # Number of batches per epoch
  num_train_trials = len(trial_lengths_train)
  epoch_size = int(np.floor(num_train_trials / cft.batch_size))
  print('Epoch size {}'.format(epoch_size))
  inputs = [train_idxs, test_idxs, ys_train, us_train, ys_test,
            ys_test_cosmooth, us_test, masks_train, masks_test,
            region_sizes, 0, 0]

  # If using true latents in simulation setting and want to track with batch
  true_latents = {}
  if xs_train is not None:
    true_latents = {
      'xs_train': xs_train,
      'xs_test': xs_test,
      'zs_train': zs_train,
      'zs_test': zs_test,
      'include_latents': True}
  extra_args = {}
  dtp = cft.dropout_trial_perc

  tf.random.set_seed(0)
  np.random.seed(0)

  # Build TF dataset loaders
  (train_dataset, test_dataset, test_dataset_cosmooth,
   train_dataset_final) = construct_tf_datasets(*inputs, dropout_trial_perc=dtp,
                                                dropout_training=cft.dropout,
                                                batch_size=cft.batch_size,
                                                random_seed=data_seed,
                                                **true_latents, **extra_args)
  train_iter = train_dataset.as_numpy_iterator()
  test_iter = test_dataset.as_numpy_iterator()
  test_batch = test_iter.next()
  test_cosmooth_iter = test_dataset_cosmooth.as_numpy_iterator()
  test_cosmooth_batch = test_cosmooth_iter.next()
  train_iter_final = train_dataset_final.as_numpy_iterator()
  train_final_batch = train_iter_final.next()
  ntrain_final = train_final_batch[0].shape[0]
  print("Loaded and prepped data.")

  eval_batches = (train_final_batch, test_batch, test_cosmooth_batch)
  if 'double-well' in cfd.data_source or 'lv' in cfd.data_source:
    eval_batches = (train_final_batch[2:], test_batch[2:], test_cosmooth_batch[2:])

  ## ----- Build model -----

  args = [num_regions, num_dims, region_sizes, trial_length]
  mrsds_model, x_transition_networks, _ = mrsds.build_model(model_dir, config_path, *args,
                                                            num_states=num_states,
                                                            zcond=cxt.zcond,
                                                            random_seed=model_seed)

  ## ----- Learning params -----

  # Regularization and annealing config
  (learning_rate_config, learning_rate_config_do,
   temperature_config, train_data_config, test_data_config) = get_configs(cft)
  objective = 'elbo'
  optimizer = tf.keras.optimizers.Adam()

  # Logging and load any existing model checkpoint for this run
  tensorboard_file_writer = tf.summary.create_file_writer(log_dir, flush_millis=100)
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=mrsds_model)
  ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=5)
  latest_checkpoint = tf.train.latest_checkpoint(model_dir)
  if latest_checkpoint:
    msg = "Loading checkpoint from {}".format(latest_checkpoint)
    ckpt.restore(latest_checkpoint)
  else:
    msg = "Starting training from scratch."
  #logging.info(msg)
  print(msg)

  # ----- Run train loop ----

  np.random.seed(training_seed)
  tf.random.set_seed(training_seed)

  # Stuff to track
  keys = ['elbos',
          'lrs',
          'temps',
          'log_pxs',
          'log_pys',
          'log_qxs',
          'train_mses',
          'train_r2s',
          'train_gen1_mses',
          'train_gen1_r2s',
          'test_mses',
          'test_r2s',
          'test_gen1_mses',
          'test_gen1_r2s',
          'cosmooth_mses',
          'cosmooth_r2s',
          'cosmooth_gen1_mses',
          'cosmooth_gen1_r2s',
          ]
  tracking = {k:[] for k in keys}

  # In mean field case, we keep training model with frozen inference net after
  # main training. This second model gets written out to a different directory
  dynamics_only = False
  if hasattr(cft, 'dynamics_only'):
    model_dir_do = result_path + name + '-models-do/'
    ckpt_manager_do = tf.train.CheckpointManager(ckpt, model_dir_do, max_to_keep=5)
    if hasattr(cft, 'mode') and cft.mode == 'mrsds_xreg_new':
      dynamics_only = True

  smooth_penalty = False
  smooth_coef = 0
  if hasattr(cft, 'smooth_coef'):
    smooth_penalty = True
    smooth_coef = cft.smooth_coef

  # We anneal beta (upweighting likelihood vs reconstruction term in elbo)
  # To make sure beta is also annealed wrt to the learning rate,
  # we use a sawtooth shape with multiple peaks
  beta = 1
  betas_anneal = False
  if hasattr(cft, 'beta_max'):
    betas_anneal = True
    from mrsds.utils_config import get_learning_rate_config
    npeaks = 10 #30 #15
    beta_config = get_learning_rate_config(
      flat_learning_rate=cft.flat_learning_rate,
      inverse_annealing_lr=cft.use_inverse_annealing_lr,
      decay_steps=cft.num_steps/npeaks,
      learning_rate=cft.beta_max,
      warmup_steps=cft.beta_warmup_steps/npeaks)
    betas = np.array([get_learning_rate(beta_config, i) for i
                      in np.arange(int(cft.num_steps))])
    betas = np.tile(betas, npeaks)

  # ---- Main training loop ----

  num_steps = cft.num_steps
  while optimizer.iterations < num_steps:

    # --- Batch setup ---
    batch = train_iter.next()
    if 'double-well' in cfd.data_source or 'lv' in cfd.data_source:
      batch = batch[2:] # Skip true xs and zs
    batch_dict = {'ys': batch[0], 'us': batch[1], 'masks': batch[2]}

    # ---- Train step ----

    current_iter = optimizer.iterations.numpy()
    learning_rate = get_learning_rate(learning_rate_config, current_iter)
    temperature = get_temperature(temperature_config, current_iter)
    if betas_anneal:
      beta = betas[current_iter]
      beta = max(beta,1)
    train_result = []  # Clear prev batch from memory

    iter_seed = training_seed + int(optimizer.iterations)

    train_result = train_step(batch_dict, mrsds_model, optimizer,
                              cft.num_samples, 0, objective, learning_rate, temperature,
                              dynamics_only=dynamics_only, smooth_penalty=smooth_penalty,
                              smooth_coef=smooth_coef, random_seed=iter_seed, beta=beta)
    # Clear from memory
    batch = []
    batch_dict = []

    # Reset iters and train dynamics only
    # For mean field case: after training,
    # we reset and start a new optimizer schedule for training dynamics only
    if (hasattr(cft, 'dynamics_only') and cft.dynamics_only
        and optimizer.iterations == num_steps-1):
      if dynamics_only:
        pass
      else:
        optimizer = tf.keras.optimizers.Adam()
        dynamics_only = True
        num_steps = cft.num_steps_do
        learning_rate_config = learning_rate_config_do
        print('dynamics only reset', optimizer.iterations)

    tracking['elbos'].append(train_result[objective].numpy())
    tracking['lrs'].append(learning_rate)
    tracking['temps'].append(temperature)
    tracking['log_pxs'].append(train_result['logpx_sum'].numpy())
    tracking['log_pys'].append(np.sum(train_result['logpy_sum'].numpy()))
    tracking['log_qxs'].append(np.sum(train_result['xt_entropy'].numpy()))

    print('iter:', current_iter, 'elbo: ', train_result[objective].numpy(),
          'logpxsum: ', train_result['logpx_sum'].numpy(),
          'lr:',learning_rate, 'beta:', beta)

    # ---- Logging ----

    if (current_iter % cft.log_steps) == 0 or current_iter == 1:

      # Evaluate model on train, test, cosmooth.
      res = eval_perf(mrsds_model, *eval_batches,
                      trial_lengths_train[:ntrain_final],
                      trial_lengths_test, dropout_idxs)

      (train_final_result, test_result, cosmooth_result,
       mse_train_final, mse_test, mse_cosmooth,
       r2_train_final, r2_test, r2_cosmooth,
       mse_gen1_train_final, mse_gen1_test, mse_gen1_cosmooth,
       r2_gen1_train_final, r2_gen1_test, r2_gen1_cosmooth) = res

      tracking['train_mses'].append(mse_train_final)
      tracking['test_mses'].append(mse_test)
      tracking['cosmooth_mses'].append(mse_cosmooth)
      tracking['train_r2s'].append(r2_train_final)
      tracking['test_r2s'].append(r2_test)
      tracking['cosmooth_r2s'].append(r2_cosmooth)

      tracking['train_gen1_mses'].append(mse_gen1_train_final)
      tracking['test_gen1_mses'].append(mse_gen1_test)
      tracking['cosmooth_gen1_mses'].append(mse_gen1_cosmooth)
      tracking['train_gen1_r2s'].append(r2_gen1_train_final)
      tracking['test_gen1_r2s'].append(r2_gen1_test)
      tracking['cosmooth_gen1_r2s'].append(r2_gen1_cosmooth)

      print(current_iter, 'mses tfr, test, cosmooth',
            mse_train_final, mse_test, mse_cosmooth)
      print(current_iter, 'mses gen1 tfr, test, cosmooth',
            mse_gen1_train_final, mse_gen1_test, mse_gen1_cosmooth)
      print(current_iter, 'r2s tfr, test, cosmooth',
            r2_train_final, r2_test, r2_cosmooth)
      print(current_iter, 'r2s gen1 tfr, test, cosmooth',
            r2_gen1_train_final, r2_gen1_test, r2_gen1_cosmooth)

    # Plotting and saving model
    if (current_iter % cft.save_steps) == 0:

      extra_args = {} #'x0_model': x0_model}
      if cxt.zcond or cft.num_samples > 1 or svae:  # don't get msgs for higher samples for now.
        extra_args = {'get_msgs': False}

      print('Saving model.')
      if dynamics_only:
        ckpt_manager_do.save()
      else:
        ckpt_manager.save()
      # Save latents etc.
      latent_region_sizes = [num_dims] * num_regions
      if hasattr(cxt.dynamics, "single_region_latent"):
        if cxt.dynamics.single_region_latent:
          latent_region_sizes = [num_dims*num_regions]
          latent_dim = np.sum(latent_region_sizes)

      # Save pre do files differently
      name_ = name
      if hasattr(cft, 'dynamics_only') and not dynamics_only:
        name_ = name.replace('-do', '')
      if hasattr(cft, 'save_training_path') and cft.save_training_path:
        name += '_{}'.format(int(current_iter / cft.save_steps))
      fpath = result_path + name_ + '_latents.mat'

      print('saving to ', fpath)

      save_results2(fpath, mrsds_model, x_transition_networks,
                   ys_train, us_train, masks_train,
                   ys_test, us_test, latent_region_sizes,
                   trial_lengths_train, trial_lengths_test,
                   *eval_batches, dropout_idxs, train_idxs, test_idxs,
                   cxt, tracking)


if __name__ == "__main__":
  """
  run with eg:
  python run_training -datapath datadir -resultpath resultdir
  -configpath path -name model1 -gpu 0 -k 2 -d 2 > logfile.txt
  """
  print(sys.argv)
  parser = argparse.ArgumentParser(description='Train a MRSDS model.')
  parser.add_argument('-datapath', type=str, required=True)
  parser.add_argument('-resultpath', type=str, required=True)
  parser.add_argument('-configpath', type=str, required=True)
  parser.add_argument('-name', type=str, required=True)
  parser.add_argument('-gpu', type=int, required=True)
  parser.add_argument('-k', type=int, required=True)
  parser.add_argument('-d', type=int, required=True)
  parser.add_argument('-mtype', type=str, default='mrsds', required=False) # Only add if yes.
  parser.add_argument('-dataseed', type=int, default=0, required=False)
  parser.add_argument('-modelseed', type=int, default=0, required=False)
  parser.add_argument('-trainseed', type=int, default=0, required=False)
  args = parser.parse_args()
  print(args)

  run_training(data_path=args.datapath, result_path=args.resultpath,
               config_path=args.configpath, name=args.name, gpu_id=args.gpu,
               num_states=args.k, num_dims=args.d, mtype=args.mtype,
               data_seed=args.dataseed, model_seed=args.modelseed,
               training_seed=args.trainseed)
