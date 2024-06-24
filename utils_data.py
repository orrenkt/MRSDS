
from itertools import product
import copy

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf

from mrsds.simulations import double_well as dw
from mrsds.simulations import lotka_volterra as lv


def load_prep_data(cfd, cft, data_path, data_seed=0, drop_region=None, max_neurons=None):

  xs = zs = task_ids = None
  print('Loading', cfd.data_source)

  if cfd.data_source == 'mydata':
    ys, us, region_sizes, num_regions = load_mydata(data_path)

  if cfd.data_source == 'mika':

    res = load_data2(data_path, trials=cfd.trials)
    (ys, us, region_sizes, num_regions, correct_idxs, error_idxs,
     towers_idxs, left_idxs, right_idxs, stages, tlim,
     excluded_idxs, orig_idxs, task_ids) = res

  elif cfd.data_source == 'double-well-2smoothnew':
    args_dict = dw_args_dict(cfd)
    #args_dict['srtype'] = "2smoothnew"  # NOTE hack to avoid new configs. fix after trying.
    args_dict['srtype'] = "2smoothnew2b"
    ret = load_data_dw(data_path, random_seed=data_seed, **args_dict)
    (ys, us, xs, zs, msgs, trial_types, choices, seq_start_idxs,
     switch_idxs, region_sizes, num_regions, C_mat1, C_mat2, ret1, ret2) = ret

  elif cfd.data_source == 'double-well-2smoothnew2b':
    args_dict = dw_args_dict(cfd)
    args_dict['srtype'] = "2smoothnew2b"
    ret = load_data_dw(data_path, random_seed=data_seed, **args_dict)
    (ys, us, xs, zs, msgs, trial_types, choices, seq_start_idxs,
     switch_idxs, region_sizes, num_regions, C_mat1, C_mat2, ret1, ret2) = ret

  elif cfd.data_source == 'lv-multiswitch':
    args_dict = lv_args_dict(cfd)
    res = load_data_lv_multiswitch(data_path, random_seed=data_seed, **args_dict)
    ys, us, xs, zs, region_sizes, num_regions, C_mat, ret = res

  elif cfd.data_source == 'lv-multiswitch2':
    args_dict = lv_args_dict(cfd)
    res = load_data_lv_multiswitch2(data_path, random_seed=data_seed, **args_dict)
    ys, us, xs, zs, region_sizes, num_regions, C_mat, ret = res

  else:
    raise ValueError('Unrecognized data source.')

  # --- Prep ---

  true_latents = {}
  if xs is not None and zs is not None:
    true_latents = {'xs': xs, 'zs': zs, 'include_latents': True}

  if max_neurons is None:
    max_neurons = np.sum(region_sizes)

  # Pad data and split into train, test, dropout for cosmoothing
  ret = prep_pad_train_test_cosmooth(ys, us, region_sizes,
                                     dropout=cft.dropout,
                                     test_perc=cft.test_perc,
                                     drop_perc=cft.drop_perc,
                                     random_seed=data_seed,
                                     maxlen=cfd.tlim,
                                     max_neurons=max_neurons,
                                     **true_latents)
  return num_regions, region_sizes, *ret


def load_mydata(data_path, random_seed=0):

  # Load your data here, using loadmat etc.
  pass
  return ys, us, xs, region_sizes, num_regions


def load_data2(fpath, trials="all", use_stages=False, region=None):
  """Load mesoscope data file."""
  dat = loadmat(fpath)
  print(dat.keys())
  if 'ys_maxnorm_low' in list(dat.keys()):
    ys = dat['ys_maxnorm_low'][()][0]
  else:
    ys = dat['ys'][()][0]
  us = dat['us'][()][0]
  region_sizes = dat['region_sizes'][()][0]
  stages = None
  if 'trial_states' in dat.keys():
    if len(dat['trial_states']) != 0:
      stages = dat['trial_states'][()][0]
      stages = [_ for _ in stages]
  num_regions = len(region_sizes)
  ys = [np.nan_to_num(_) for _ in ys]
  us = [_.astype(np.float32) for _ in us]
  print('Loaded {} trials, {} regions. Num neurons: {}. First trial length: {}. Input dims: {}'
         .format(len(ys), num_regions, region_sizes, ys[0].shape, us[0].shape))
  correct_idxs = dat['correct_idxs'][()][0] - 1
  error_idxs = []
  if len(dat['error_idxs']) != 0:
    error_idxs = dat['error_idxs'][()][0] - 1
  towers_idxs = dat['towers_idxs'][()][0] - 1
  visual_idxs = np.array([i for i in range(len(ys)) if i not in towers_idxs])
  left_idxs = dat['left_idxs'][()][0] - 1
  right_idxs = dat['right_idxs'][()][0] - 1

  # Optionally concat trial stage into inputs
  if use_stages:
    us_ = []
    for i in range(len(us)):
      us_.append(np.hstack([us[i], stages[i].T]))
    us = us_

  # Optionally select subset of trial types
  if trials == 'towers':
    us = filter_list(us, towers_idxs)
    ys = filter_list(ys, towers_idxs)
    stages = filter_list(stages, towers_idxs)
  elif trials == 'visual':
    us = filter_list(us, towers_idxs, 'not')
    ys = filter_list(ys, towers_idxs, 'not')
    stages = filter_list(stages, towers_idxs, 'not')
  elif trials == 'small':
    us = us[:32]
    ys = ys[:32]
    stages = stages[:32]
  trial_lengths = [y.shape[0] for y in ys]
  tlim = np.max(trial_lengths)
  print('ys', len(ys))

  # NOTE  most trials are much shorter than max. throwing out over 185.
  tlim = 185

  # Optionally exclude long trials
  j = 0
  orig_idxs = []
  task_ids = []
  if tlim is not None:
    excluded_idxs = []
    ys_ = []
    us_ = []
    left_idxs_ = []
    right_idxs_ = []
    correct_idxs_ = []
    error_idxs_ = []
    stages_ = []

    for i, (y, u) in enumerate(zip(ys, us)):
      trial_length = y.shape[0]
      if trial_length <= tlim:
        ys_.append(np.nan_to_num(y))
        us_.append(u)
        if i in left_idxs:
          left_idxs_.append(j)
        elif i in right_idxs:
          right_idxs_.append(j)
        if i in correct_idxs:
          correct_idxs_.append(j)
        else:
          error_idxs_.append(j)
        stages_.append(stages[i])
        orig_idxs.append(i)
        task_id = 1 if i in towers_idxs else 2
        task_ids.append(task_id)
        j += 1
      else:
        excluded_idxs.append(i)
    # NOTE should log instead.
    print('Dropped {} trials due to length > {}'.format(len(excluded_idxs), tlim))
    ys = ys_
    us = us_
    left_idxs = left_idxs_
    right_idxs = right_idxs_
    correct_idxs = correct_idxs_
    error_idxs = error_idxs_
    stages = stages_
    trial_lengths = [y.shape[0] for y in ys]
    tlim = np.max(trial_lengths)
    task_ids = np.array(task_ids)[:,np.newaxis].astype(np.float32)

  if region is not None:
    num_regions = 1
    start, end = np.cumsum([0, *region_sizes])[region:region+2]
    ys = [_[:,start:end] for _ in ys]
    region_sizes = [region_sizes[region]]
    print('single region', region, region_sizes)

  return (ys, us, region_sizes, num_regions, correct_idxs, error_idxs,
          towers_idxs, left_idxs, right_idxs, stages, tlim,
          excluded_idxs, orig_idxs, task_ids)


def load_data_dw(fpath, T=75, r1s2_noise=0.005,
                 num_seqs=30, evidence_period=[4,30], #60,30
                 num_trials=15, plot=False, obs_dims=100, #50,
                 maxnorm_obs=False, srtype=1, accum_only=False,
                 split_obs=False, random_seed=0, history_effect=False,
                 x1_ranges=None, dx1s=None, x2_ranges=None, dx2s=None,
                 nle=False):
  """TBD."""
  np.random.seed(random_seed)

  if x1_ranges is None:
    #x1, x2 = dw.generate_grid([-1.5,1.6],[-0.1,0.7], 0.15, 0.05) #0.1, 0.05)
    x1_1, x2_1 = dw.generate_grid([-1.25,1.3],[-0.05,0.55], 0.1, 0.035) #0.1, 0.05)
    #x1_2, x2_2 = dw.generate_grid([-2,2],[-2,2], 0.2, 0.25)
    x1_2, x2_2 = dw.generate_grid([-1,1],[-1,1], 0.1, 0.1)

  else:
    x1_1, x2_1 = dw.generate_grid(x1_ranges[0], x2_ranges[0], dx1s[0], dx2s[0])
    x1_2, x2_2 = dw.generate_grid(x1_ranges[1], x2_ranges[1], dx1s[1], dx2s[1])

  bias = 1  # No intrinsic bias in R1 dynamics.
  r2_noise = 0.055
  r2_scale = 0.1 * np.array([0.05, 0.3])  #4.5 0.2
  r1_scale = 0.1
  feedback = True
  history = history_effect

  # We generate multiple sequences of trials
  # Eg 10 total x 15 trials = 250.
  total = num_seqs*num_trials
  seq_start_idxs = list(np.arange(0, total, num_trials))
  seq_seed = 0  # we'll modify this for each sequence, used to be fixed.

  # Observations
  num_regions = 2
  obs_seed = 0  # Same across sequences of trials

  xs_trials = []
  us_trials = []
  ys_trials = []
  zs_trials = []
  msgs_trials = []
  type_trials = []
  choice_trials = []
  all_switch_idxs = []

  sim_args = {
    'feedback': feedback,
    'r1s2_noise': r1s2_noise,
    'r1_scale': r1_scale,
    'history_effect': history,
    'accum_only': accum_only
  }
  print(sim_args)

  print('srtype', srtype)

  for i in range(num_seqs):

    seq_seed = random_seed + i

    # Generate trial stimuli
    us, trial_types = dw.gen_trial_inputs(num_trials, T=T, ratio=0.5,
                                          #means=[[5,30],[30,5]], # bigger diff
                                          means=[[5,20],[20,5]], # bigger diff
                                          evidence_period=[3,20], # shorter evidence period
                                          random_seed=seq_seed)

    # Simulate trial latents and observations
    if srtype == 2:
      xs1, dt1, msgs, switch_idxs = dw.simulate_xs_mr2(us, T, **sim_args)
    elif srtype == "2smooth":
      xs1, dt1, msgs, switch_idxs = dw.simulate_xs_mr2_smooth(us, T, **sim_args)
    elif srtype == "2smoothnew":
      xs1, xs2, dt1, msgs, switch_idxs = dw.simulate_xs_mr2_smoothnew(us, T, **sim_args)
    elif srtype == "2smoothnew2":
      xs1, xs2, dt1, msgs, switch_idxs = dw.simulate_xs_mr2_smoothnew2(us, T, **sim_args,
                                                                       random_seed=seq_seed)
    elif srtype == "2smoothnew2b":
      xs1, xs2, dt1, msgs, switch_idxs = dw.simulate_xs_mr2_smoothnew2b(us, T, **sim_args,
                                                                       random_seed=seq_seed)
    else:
      xs1, dt1, msgs, switch_idxs = dw.simulate_xs_mr(us, T, **sim_args)

    zs = np.zeros([num_trials, xs1[0].shape[0]]).astype(int)
    for tr, switch_idx in enumerate(switch_idxs):
      zs[tr,switch_idx:] += 1
    all_switch_idxs.extend(switch_idxs)

    # With input
    us_ = np.array(us)
    for tr, switch_idx in enumerate(switch_idxs):
      us_[tr,switch_idx-1,-1] += 1
    us = list(us_)

    #xs = [np.hstack([_1.T, _2.T]) for _1, _2 in zip(xs1, xs2)]
    xs = np.concatenate([np.array(xs1), np.array(xs2)], axis=-1)

    if nle:
      ys = dw.generate_observations_mr_nle(xs, obs_dims, num_regions, obs_seed)
      C_mat1 = C_mat2 = None
    else:
      ys, C_mat1, C_mat2 = dw.generate_observations_mr(xs, obs_dims,
                                                       num_regions, obs_seed)
    xs_trials.extend(xs)
    us_trials.extend(us)
    ys_trials.extend(ys)
    zs_trials.extend(zs)
    msgs_trials.extend(msgs)
    type_trials.extend(trial_types)

    signs = [x1_[switch_idx,0] for x1_, switch_idx in zip(xs1, switch_idxs)]
    choices = [-1 if sign<=0 else 1 for sign in signs]
    choice_trials.extend(choices)
    seq_seed += 1

    if srtype == 2:
      ret1 = dw.get_potential_grad2(x1_1, x2_1, bias)
      ret2 = dw.get_potential_grad_r2(x1_2, x2_2)
    if srtype == "2smooth": # in srtype:
      ret1 = dw.get_potential_grad2_smooth(x1_1, x2_1, bias)
      ret2 = dw.get_potential_grad_r2(x1_2, x2_2)
    else:
      ret1 = dw.get_potential_grad(x1_1, x2_1, bias)
      ret2 = dw.get_potential_grad_r2(x1_2, x2_2)

    if plot:
      title = ''
      inputs = [x1_1, x2_1, x1_2, x2_2, *ret1, *ret2, xs1, xs2, zs,
                T, dt1, dt1, title, switch_idxs, trial_types, msgs, xs]
      dw.visualize_trajectories(*inputs, ntrials=num_trials)

  max_neurons = ys[0].shape[-1]
  region_sizes = [int(max_neurons/2), int(max_neurons/2)]
  num_regions = 2

  return (ys_trials, us_trials, xs_trials, zs_trials, msgs_trials, type_trials,
          choice_trials, seq_start_idxs, all_switch_idxs, region_sizes, num_regions,
          C_mat1, C_mat2, (x1_1, x2_1, *ret1), (x1_2, x2_2, *ret2))


def load_data_lv_multiswitch2(fpath, T=75000, sample=500, dt=0.001, batch_size=32,
                 region_sizes=[50,50], # NOTE switched post iclr from [30,30], #[30,30], #[15,15],
                 latent_dims=2, num_trials=500, #100, #288, 200
                 x1_range=[0.25, 2.5], x2_range=[0.25,2], dx1=0.125, dx2=0.125,
                 noise_level=1e-3, regime=None, loc_switch=False,
                 zswitch=False, absmaxnorm=False, random_seed=0,
                 lv_version=2, switch_signal=True, uz=False, random_zic=False,
                 nl_emissions=False, nl_emissions_type='gaussian_bump'):

  np.random.seed(random_seed)

  alpha = 2.0 / 3.0
  beta = 4.0 / 3.0
  gamma = 1.0
  delta = 1.0
  params1 = [alpha, beta, gamma, delta]

  alpha2 = 0.9
  beta2 = 1.1
  gamma2 = 0.8
  delta2 = 1.2

  params2 = [alpha2, beta2, gamma2, delta2]

  # Fixed points
  fp1 = (gamma / delta, alpha / beta)
  fp2 = (gamma2 / delta2, alpha2 / beta2)

  x0_base = 1.0 #0.4
  y0_base = 1.0 #0.4

  obs_dim = np.sum(region_sizes)
  num_regions = len(region_sizes)
  region_sizes = [int(obs_dim/2), int(obs_dim/2)]

  ys_trials = []
  xs_trials = []
  us_trials = []
  zs_trials = []

  C_mat = np.random.randn(obs_dim, latent_dims)
  C_mat[:region_sizes[0], 0] *= 0.0
  C_mat[region_sizes[0]:, 1] *= 0.0

  for nt in range(num_trials):
    #x0 = x0_base + 1.0 * np.random.rand()
    #y0 = y0_base + 1.0 * np.random.rand()
    x0 = x0_base + 0.2 * np.random.rand()
    y0 = y0_base + 0.2 * np.random.rand()

    nswitches = np.random.choice([1,2,3])
    trange = np.arange(5000, 70500, 500)
    tswitches = list(np.sort(np.random.choice(trange, nswitches)))
    x0 = np.array([x0, y0])
    seed = random_seed + nt
    x, y, z = lv.generate_switching_lotka_volterra_multiswitch2(C_mat, tswitches=tswitches,
                                                               params=[params1,params2], T=T,
                                                               sample=sample, regime=regime,
                                                               latent_dims=latent_dims,
                                                               obs_dims=obs_dim, dt=dt, x0=x0,
                                                               noise_level=noise_level,
                                                               dtype=np.float32, return_latent=True,
                                                               random_zic=random_zic, random_seed=seed,
                                                               nl_emissions=nl_emissions,
                                                               nl_emissions_type=nl_emissions_type)

    #u = np.zeros([int(T/sample), 2])
    u = np.zeros([x.shape[0], 2])
    if switch_signal and regime is None:
      switchpts = np.where(z[:-1]!=z[1:])[0] #-1 # NOTE -1 for DEBUG! # + 1
      #switchpt = np.where(z[1:]!=z[:-1])[0][0] + 1
      for switchpt in switchpts:
        u[switchpt,0] = 1

    if uz:
      #u = np.zeros([int(T/sample), 2])
      u = np.zeros([x.shape[0], 2])
      u[:-1,0] = z[1:]
      #u[:,1] = z

    xs_trials.append(x)
    ys_trials.append(y)
    us_trials.append(u)
    zs_trials.append(z)

  x1_ = np.arange(*x1_range, dx1)
  x2_ = np.arange(*x2_range, dx2)
  x1_, x2_ = np.meshgrid(x1_, x2_)

  ret = [x1_, x2_, *lv.get_potential_grad(x1_, x2_, params1, params2), fp1, fp2]

  def maxnorm(x):
    return x / np.max(np.abs(x))

  if absmaxnorm:
    ys_flat = np.array(ys_trials).reshape(num_trials*x.shape[0], obs_dim)
    shapes = (num_trials, x.shape[0], obs_dim)
    ys_trials = list(np.apply_along_axis(maxnorm, arr=ys_flat, axis=0).reshape(*shapes))

  return ys_trials, us_trials, xs_trials, zs_trials, region_sizes, num_regions, C_mat, ret



def load_data_lv_multiswitch(fpath, T=75000, sample=500, dt=0.001, batch_size=32,
                 region_sizes=[50,50], # NOTE switched post iclr from [30,30], #[30,30], #[15,15],
                 latent_dims=2, num_trials=500, #100, #288, 200
                 x1_range=[0.25, 2.5], x2_range=[0.25,2], dx1=0.125, dx2=0.125,
                 noise_level=1e-3, regime=None, loc_switch=False,
                 zswitch=False, absmaxnorm=False, random_seed=0,
                 lv_version=2, switch_signal=True, uz=False, random_zic=False, nl_emissions=False):

  #if lv_version == 3:
  #  import mrsds.simulations.lotka_volterra3 as lv
  #else:
  #  import mrsds.simulations.lotka_volterra2 as lv

  np.random.seed(random_seed)

  alpha = 2.0 / 3.0
  beta = 4.0 / 3.0
  gamma = 1.0
  delta = 1.0
  params1 = [alpha, beta, gamma, delta]

  alpha2 = 0.9
  beta2 = 1.1
  gamma2 = 0.8
  delta2 = 1.2

  params2 = [alpha2, beta2, gamma2, delta2]

  # Fixed points
  fp1 = (gamma / delta, alpha / beta)
  fp2 = (gamma2 / delta2, alpha2 / beta2)

  x0_base = 1.0 #0.4
  y0_base = 1.0 #0.4

  obs_dim = np.sum(region_sizes)
  num_regions = len(region_sizes)
  region_sizes = [int(obs_dim/2), int(obs_dim/2)]

  ys_trials = []
  xs_trials = []
  us_trials = []
  zs_trials = []

  C_mat = np.random.randn(obs_dim, latent_dims)
  C_mat[:region_sizes[0], 0] *= 0.0
  C_mat[region_sizes[0]:, 1] *= 0.0

  for nt in range(num_trials):
    #x0 = x0_base + 1.0 * np.random.rand()
    #y0 = y0_base + 1.0 * np.random.rand()
    x0 = x0_base + 0.2 * np.random.rand()
    y0 = y0_base + 0.2 * np.random.rand()

    nswitches = np.random.choice([1,2,3])
    trange = np.arange(5000, 70500, 500)
    tswitches = list(np.sort(np.random.choice(trange, nswitches)))
    x0 = np.array([x0, y0])
    seed = random_seed + nt
    x, y, z = lv.generate_switching_lotka_volterra_multiswitch(C_mat, tswitches=tswitches,
                                                               params=[params1,params2], T=T,
                                                               sample=sample, regime=regime,
                                                               latent_dims=latent_dims,
                                                               obs_dims=obs_dim, dt=dt, x0=x0,
                                                               noise_level=noise_level,
                                                               dtype=np.float32, return_latent=True,
                                                               random_zic=random_zic, random_seed=seed,
                                                               nl_emissions=nl_emissions)

    #u = np.zeros([int(T/sample), 2])
    u = np.zeros([x.shape[0], 2])
    if switch_signal and regime is None:
      switchpts = np.where(z[:-1]!=z[1:])[0] #-1 # NOTE -1 for DEBUG! # + 1
      #switchpt = np.where(z[1:]!=z[:-1])[0][0] + 1
      for switchpt in switchpts:
        u[switchpt,0] = 1

    if uz:
      #u = np.zeros([int(T/sample), 2])
      u = np.zeros([x.shape[0], 2])
      u[:-1,0] = z[1:]
      #u[:,1] = z

    xs_trials.append(x)
    ys_trials.append(y)
    us_trials.append(u)
    zs_trials.append(z)

  x1_ = np.arange(*x1_range, dx1)
  x2_ = np.arange(*x2_range, dx2)
  x1_, x2_ = np.meshgrid(x1_, x2_)

  ret = [x1_, x2_, *lv.get_potential_grad(x1_, x2_, params1, params2), fp1, fp2]

  def maxnorm(x):
    return x / np.max(np.abs(x))

  if absmaxnorm:
    ys_flat = np.array(ys_trials).reshape(num_trials*x.shape[0], obs_dim)
    shapes = (num_trials, x.shape[0], obs_dim)
    ys_trials = list(np.apply_along_axis(maxnorm, arr=ys_flat, axis=0).reshape(*shapes))

  return ys_trials, us_trials, xs_trials, zs_trials, region_sizes, num_regions, C_mat, ret


def prep_pad_train_test_cosmooth(ys, us, region_sizes, test_perc=0.1, drop_perc=0.25,
                                 random_seed=0, maxlen=None, max_neurons=None, xs=None,
                                 zs=None, include_latents=False,
                                 seq_start_idxs=None, dropout=False):
  """
  Pad trials to same length, split into train/test and dropout neurons for cosmoothing.
  """
  # Zero pad trials ys and us to standardize trial length
  trial_lengths = [_.shape[0] for _ in ys]
  if maxlen is None:
    maxlen = np.max(trial_lengths)
  if max_neurons is None:
    max_neurons = np.sum(region_sizes)
  neurons_diff = max_neurons - np.sum(region_sizes)
  print(maxlen, len(ys), ys[0].shape, neurons_diff, max_neurons)
  ys_padded = [np.pad(_, ((0,maxlen-_.shape[0]),(0,neurons_diff)),
                 'constant', constant_values=0) for _ in ys]
  us_padded = [np.pad(_, ((0,maxlen-_.shape[0]),(0,0)),
               'constant', constant_values=0) for _ in us]

  # Create bool mask for padding timepoints
  num_trials = len(trial_lengths)
  masks = np.full((num_trials, maxlen, max_neurons), True, dtype=bool)
  for i, tl in enumerate(trial_lengths):
    masks[i,tl:,:] = False
  if neurons_diff > 0:
    masks[:,:,-neurons_diff:] = False

  # Set seed
  np.random.seed(random_seed)

  # Split into train and test trials
  num_trials = len(ys_padded)
  ntrain = int(num_trials*(1-test_perc))
  train_idxs = np.random.choice(np.arange(num_trials), ntrain, replace=False)
  test_idxs = np.array([_ for _ in np.arange(num_trials) if _ not in train_idxs])

  ys_padded_train = filter_list(ys_padded, train_idxs)
  us_padded_train = filter_list(us_padded, train_idxs)
  ys_padded_test = filter_list(ys_padded, test_idxs)
  us_padded_test = filter_list(us_padded, test_idxs)
  masks_train = filter_list(masks, train_idxs)
  masks_test = filter_list(masks, test_idxs)
  trial_lengths_train = filter_list(trial_lengths, train_idxs)
  trial_lengths_test = filter_list(trial_lengths, test_idxs)
  print('Train trials: {}, test trials: {}.'.format(train_idxs.shape, test_idxs.shape))
  print('test indices', test_idxs)

  region_sizes_cumsum = np.cumsum(np.stack([0, *region_sizes]))
  starts_ends = list(zip(region_sizes_cumsum, region_sizes_cumsum[1:]))

  # Generate random drop indices
  rnd_orderings = []
  drop_end_idxs = []
  for i, (start, end) in enumerate(starts_ends):
    num_neurons = end-start
    nrange = np.arange(start, end)
    rnd_ordering = np.random.permutation(nrange)
    rnd_orderings.append(rnd_ordering)
    drop_end_idx = int(num_neurons*drop_perc)
    drop_end_idxs.append(drop_end_idx)

  # Mask dropped neurons
  ys_padded_test_cosmooth = np.asarray(ys_padded_test)
  if dropout:
    dropout_idxs = np.concatenate([rnd_orderings[j][:drop_end_idxs[j]]
                                    for j in range(len(region_sizes))])
    print('Dropping {} neurons.'.format(len(dropout_idxs)))
    print('Using drop percent', drop_perc)
    print(dropout_idxs)
    print(ys_padded_test_cosmooth.shape)
    ys_padded_test_cosmooth[:, :, dropout_idxs] = 0
  else:
    dropout_idxs = []
  ys_padded_test_cosmooth = list(ys_padded_test_cosmooth)

  # When using with a simulator
  _ = []
  if xs is not None and zs is not None and include_latents is True:
    xs_train = filter_list(xs, train_idxs)
    xs_test = filter_list(xs, test_idxs)
    zs_train = filter_list(zs, train_idxs)
    zs_test = filter_list(zs, test_idxs)
  else:
    xs_train = xs_test = zs_train = zs_test = None
  _.extend([xs_train, xs_test, zs_train, zs_test])

  return (ys_padded_train, us_padded_train, ys_padded_test, us_padded_test,
          ys_padded_test_cosmooth, dropout_idxs, trial_lengths_train, trial_lengths_test,
          train_idxs, test_idxs, masks_train, masks_test, *_)


def generate_dropout_masks2(region_sizes, num_neurons, num_trials,
                            dropout_perc=0.25, multiple=10):
  """
  Generate random dropout masks, to be shuffled seperately from data on each epoch.
  All the masks include dropout, we sample from trial_perc in the train loop and include if true
  Otherwise add generic null mask.
  Multiday ys can be padded ie num_neurons>sum(region_sizes).
  Only dropout neurons based on region_sizes
  """
  print(region_sizes)
  # TODO: handle multiday neuron padding
  region_sizes_cumsum = np.cumsum([0] + list(region_sizes))
  nt = multiple*num_trials  # Generate x50 to get diversity
  masks = np.ones([nt, num_neurons]).astype(bool)
  all_drop_idxs = []
  for idx in range(nt):
    drop_idxs = []
    # Per region dropout
    for i, region_size in enumerate(region_sizes):
      drop_ = int(region_size*dropout_perc)
      nidxs = np.random.choice(np.arange(*region_sizes_cumsum[i:i+2]),
                               drop_, replace=False)
      masks[idx, nidxs] = False
      drop_idxs.extend(nidxs)
    all_drop_idxs.append(drop_idxs)
  return masks.astype(np.float32), all_drop_idxs


def construct_tf_datasets2(train_idxs, test_idxs, ys_train, us_train, ys_test,
                           ys_test_cosmooth, us_test, masks_train, masks_test,
                           #task_ids_train, task_ids_test,
                           region_sizes, day_id=0, animal_id=0, ys_history_train=None,
                           ys_history_test=None, dropout_training=False, dropout_perc=0.25,
                           dropout_vary=False, vary_perc=0.5, dropout_trial_perc=1,
                           xs_train=None, xs_test=None, zs_train=None, zs_test=None,
                           include_latents=False, include_history=False, mask_multiple=10,
                           random_seed=0, batch_size=64):

  # TODO: handle multiday neuron padding

  if ys_history_train is None:
    print('No ys history provided, using dummy')
    ys_history_train = np.zeros((len(ys_train),1))
    ys_history_test = np.zeros((len(ys_test),1))

  tf.random.set_seed(random_seed)
  np.random.seed(random_seed)

  # When using a simulator we optionally add true latent to each batch
  train_latents = []
  test_latents = [[],[]]
  if include_latents:
    train_latents.extend([tf.data.Dataset.from_tensor_slices(_)
                          for _ in (xs_train, zs_train)])
    test_latents[0].extend([np.as_array(xs_test), np.asarray(zs_test)])
    test_latents[1].extend([np.asarray(xs_train[:batch_size]),
                            np.asarray(zs_train[:batch_size])])

  # Create zipped tf train dataset with all tensors. Batch later to combine with masks.
  datasets = [*train_latents]
  day_ids = np.zeros([len(ys_train),1]).astype(np.int32) + day_id
  animal_ids = np.zeros([len(ys_train),1]).astype(np.int32) + animal_id
  for array in (ys_train, us_train, day_ids, animal_ids, ys_history_train):
    datasets.append(np.asarray(array))
  train_dataset = tf.data.Dataset.from_tensor_slices(tuple(datasets))

  # Add mask dataset, to be shuffled seperately. Not batched.
  if dropout_training:
    masks, drop_idxs = generate_dropout_masks2(region_sizes, ys_train[0].shape[-1],
                                               len(ys_train), dropout_perc, mask_multiple)
    train_masks = tf.data.Dataset.from_tensor_slices(masks)
    train_dropids = tf.data.Dataset.from_tensor_slices(drop_idxs)
    train_masks = tf.data.Dataset.zip((train_masks, train_dropids))
  else:
    masks = masks_train
    train_masks = tf.data.Dataset.from_tensor_slices(masks)

  ntest = len(ys_test)
  repeats = int(np.ceil(batch_size / ntest))
  td = (*test_latents[0], np.asarray(ys_test), np.asarray(us_test),
        np.asarray(masks_test).astype(np.float32), np.repeat(day_id, ntest),
        np.repeat(animal_id, ntest), np.asarray(ys_history_test))
  test_dataset = (tf.data.Dataset.from_tensor_slices(td)
                  .repeat(repeats).batch(batch_size,drop_remainder=True))
  print('test size', ntest, batch_size, td[0].shape, td[1].shape) #, tf.data.Dataset.from_tensor_slices(td).repeat(2))

  # For cosmoothing
  td_ = (*test_latents[0], np.asarray(ys_test_cosmooth), np.asarray(us_test),
         np.asarray(masks_test).astype(np.float32), np.repeat(day_id, ntest),
         np.repeat(animal_id, ntest), np.asarray(ys_history_test))
  test_dataset_cosmooth = (tf.data.Dataset.from_tensor_slices(td_)
                           .repeat(repeats).batch(batch_size,drop_remainder=True))
  print('test cosmooth size', batch_size, td_[0].shape, td_[1].shape) #, tf.data.Dataset.from_tensor_slices(td).repeat(2))

  # TODO: double check that the masks for final batch are fine

  # We'll eventually evaluate on a single trained batch
  td2 = (*test_latents[1], np.asarray(ys_train[:batch_size]),
         np.asarray(us_train[:batch_size]), np.asarray(masks_train[:batch_size]).astype(np.float32),
         np.repeat(day_id, batch_size), np.repeat(animal_id, batch_size),
         np.asarray(ys_history_train[:batch_size]))
  print('train final size', batch_size, td2[0].shape, td2[1].shape)
  train_dataset_final = (tf.data.Dataset.from_tensor_slices(td2)
                         .batch(batch_size=batch_size))

  return (train_dataset, train_masks, test_dataset, test_dataset_cosmooth,
          train_dataset_final)


def construct_tf_datasets(train_idxs, test_idxs, ys_train, us_train, ys_test,
                          ys_test_cosmooth, us_test, masks_train, masks_test,
                          region_sizes, day_id=0, animal_id=0, ys_history_train=None,
                          ys_history_test=None, dropout_training=False, dropout_perc=0.25,
                          dropout_vary=False, vary_perc=0.5, dropout_trial_perc=1,
                          dropout_type="full", #alt: "neurons", # alt: full
                          xs_train=None, xs_test=None, zs_train=None, zs_test=None,
                          include_latents=False, include_history=False,
                          random_seed=0, batch_size=64):

  if ys_history_train is None:
    ys_history_train = np.zeros((len(ys_train),1))
    ys_history_test = np.zeros((len(ys_test),1))

  tf.random.set_seed(random_seed)
  np.random.seed(random_seed)

  # Multiday ys can be padded, so only dropout neurons based on region_sizes
  if dropout_training:
    print('dropout training', dropout_type, dropout_trial_perc, dropout_perc)
    num_neurons = np.sum(region_sizes)
    drop_num = int(num_neurons*dropout_perc)
    neuron_idxs = np.arange(num_neurons)

  def generator():
    while True:
      # Pick trial index
      idx = np.random.choice(np.arange(len(train_idxs)))
      _ = []
      if include_latents:  # when using a simulator
        _.append(xs_train[idx].astype(np.float32)) #32
        _.append(zs_train[idx])
      if not dropout_training:
        yield (*_, ys_train[idx], us_train[idx],
               masks_train[idx].astype(np.float32))
      else:
        mask = np.copy(masks_train[idx])
        drop_idxs = np.zeros(drop_num) - 1  # -1 for None
        # If we don't drop out all trials
        if np.random.random() <= dropout_trial_perc:
          if dropout_type == "neurons":
            drop_idxs = np.random.choice(neuron_idxs, drop_num, replace=False)
            if dropout_vary and np.random.random() <= vary_perc:
              drop_num_ = np.random.choice(np.arange(drop_num))
              mask[:,drop_idxs[:drop_num_]] = False
              drop_idxs[drop_num_:] = -1
            else:
              mask[:,drop_idxs] = False
          elif dropout_type == "full":
            mask = np.random.binomial(1, dropout_perc, size=mask.shape).astype(bool)
          else:
            raise ValueError("Dropout type unsupported.")

        yield (*_, ys_train[idx].astype(np.float32),
               us_train[idx].astype(np.float32), mask.astype(np.float32))

  ntest = len(test_idxs)

  # When using a simulator we optionally add true latent to each batch
  _ = [[],[],[],[],[]]
  if include_latents:
    _[0].append(tf.float32) #32)
    _[0].append(tf.int32)
    _[1].append(np.asarray(xs_test))
    _[2].append(np.asarray(xs_train[:ntest]))
    _[3].append(np.asarray(zs_test))
    _[4].append(np.asarray(zs_train[:ntest]))

  types = (*_[0], tf.float32, tf.float32, tf.float32)
  dataset = (tf.data.Dataset.from_generator(generator, output_types=types)
             .batch(batch_size=batch_size)
             .cache()
             .prefetch(tf.data.experimental.AUTOTUNE)
             .repeat(count=-1)) # repeat infinitely

  td = (*_[1], *_[3], np.asarray(ys_test).astype(np.float32),
        np.asarray(us_test).astype(np.float32),
        np.asarray(masks_test).astype(np.float32))
  test_dataset = (tf.data.Dataset.from_tensor_slices(td)
                  .batch(batch_size=ntest))

  # For cosmoothing
  td_ = (*_[1], *_[3], np.asarray(ys_test_cosmooth).astype(np.float32),
         np.asarray(us_test).astype(np.float32),
         np.asarray(masks_test).astype(np.float32))
  test_dataset_cosmooth = (tf.data.Dataset.from_tensor_slices(td_)
                           .batch(batch_size=ntest))

  # We'll eventually evaluate on a single trained batch
  td2 = (*_[2], *_[4], np.asarray(ys_train[:ntest]).astype(np.float32),
         np.asarray(us_train[:ntest]).astype(np.float32),
         np.asarray(masks_train[:ntest]).astype(np.float32))
  train_dataset_final = (tf.data.Dataset.from_tensor_slices(td2)
                         .batch(batch_size=len(test_idxs)))

  return dataset, test_dataset, test_dataset_cosmooth, train_dataset_final


def filter_list(datalist, idxs, ftype=None):
  if ftype == 'not':
    return [_ for i, _ in enumerate(datalist) if i not in idxs]
  else:
    return [_ for i, _ in enumerate(datalist) if i in idxs]


def dw_args_dict(cfd):
  accum_only = False
  split_obs = False
  if hasattr(cfd, 'accum_only'):
    accum_only = cfd.accum_only
  if hasattr(cfd, 'split_obs'):
    split_obs = cfd.split_obs
  args_dict = {'accum_only': accum_only,
               'split_obs': split_obs}
  if hasattr(cfd, 'num_seqs'):
    args_dict['num_seqs'] = cfd.num_seqs
  if hasattr(cfd, 'history_effect'):
    args_dict['history_effect'] = cfd.history_effect
  if hasattr(cfd, 'xswitch') and cfd.xswitch:
    args_dict['xswitch'] = cfd.xswitch
  if hasattr(cfd, 'nle') and cfd.nle:
    args_dict['nle'] = cfd.nle
  return args_dict


def lv_args_dict(cfd):
  args_dict = {}
  if hasattr(cfd, 'regime') and cfd.regime is not None:
    args_dict['regime'] = int(cfd.regime)
  if hasattr(cfd, 'loc_switch') and cfd.loc_switch:
    args_dict['loc_switch'] = int(cfd.loc_switch)
  if hasattr(cfd, 'zswitch') and cfd.zswitch:
    args_dict['zswitch'] = int(cfd.zswitch)
  if hasattr(cfd, 'absmaxnorm') and cfd.absmaxnorm:
    args_dict['absmaxnorm'] = int(cfd.absmaxnorm)
  if hasattr(cfd, 'num_trials'):
    args_dict['num_trials'] = int(cfd.num_trials)
  else:
    print('Num trials not provided in config, using default.')
  if hasattr(cfd, 'random_zic'):
    args_dict['random_zic'] = bool(cfd.random_zic)
  if hasattr(cfd, 'uz'):
    args_dict['uz'] = bool(cfd.uz)
  if hasattr(cfd, 'nle'):
    args_dict['nl_emissions'] = bool(cfd.nle)
  if hasattr(cfd, 'nle_type'):
    args_dict['nl_emissions_type'] = bool(cfd.nle_type)
  return args_dict
