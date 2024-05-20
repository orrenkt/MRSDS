
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA #, ProbabilisticPCA
from sklearn.metrics import r2_score as sk_r2_score

from mrsds.utils_tensor import normalize_logprob


def get_communication_models(x_transition_networks, num_regions, num_states, input_len=0):
  """Returns N(N+input_len) communication models."""
  communication_models = [[[] for _ in range(num_regions)]
                          for _ in range(num_states)]
  for k in range(num_states):
    for i in range(num_regions):
      for j in range(num_regions):
        output_layer = x_transition_networks[k].get_layer("{}-{}".format(i,j))
        # NOTE: was this an error indexing on 0 or needed for non svae?
        #m = Model(inputs=x_transition_networks[k].inputs[0], outputs=[output_layer.output])
        m = Model(inputs=x_transition_networks[k].inputs, outputs=[output_layer.output])
        communication_models[k][i].append(m)
      # External inputs (us), one comm model per dimension
      for l in range(input_len):
        output_layer = x_transition_networks[k].get_layer("{}-u-{}".format(i, l))
        m = Model(inputs=x_transition_networks[k].inputs, outputs=[output_layer.output])
        communication_models[k][i].append(m)
  return communication_models


def get_msgs_and_norms_svae(communication_models, result_dict, num_states, num_regions, latent_dim,
                            latent_region_sizes, trial_lengths, input_len, input_regions='all',
                            conditional_msgs=False, conditional_inputs=False):
  """
  Note locally conditioned model requires concatenating local and second region xs.
  Also note input models takes in entire x vector across regions and split region internally.
  """
  xs_input = result_dict['x_sampled'].numpy()[:,:-1,:]
  if len(xs_input.shape) == 4 and xs_input.shape[-2] == 1: # otherwise breaks for d=1 case.
    xs_input = np.squeeze(xs_input,axis=-2)
  us_input = result_dict['us'].numpy()[:,1:,:]
  num_trials, num_timepoints = xs_input.shape[0:2]

  latent_region_sizes_cs = np.cumsum([0] + latent_region_sizes)
  starts_ends = list(zip(latent_region_sizes_cs[:-1], latent_region_sizes_cs[1:]))

  # Create copies of xs input with each region zeroed out
  if conditional_msgs:
    xs_input0s = []
    for i in range(num_regions):
      start, end = starts_ends[i]
      xs_input0 = np.copy(xs_input)
      xs_input0[:,:,start:end] = 0
      xs_input0s.append(xs_input0)

  if conditional_inputs:
    us_input0s = []
    for l in range(input_len):
      us_input0 = np.copy(us_input)
      us_input0[:,:,l] = 0
      us_input0s.append(us_input0)

  #if zcond:
  #  zs = []
  #  for i in range(num_states):
  #    zs1 = np.zeros([num_trials, num_timepoints, num_states])
  #    zs1[:,:,i] = 1
  #  zs.append(zs1)

  # Get messages
  messages = np.zeros((num_states, num_trials, num_timepoints+1,
                       num_regions, num_regions+input_len, latent_dim))
  messages_diff = np.zeros_like(messages)
  for k in range(num_states):
    for i in range(num_regions):
      start, end = starts_ends[i]
      xs_i = xs_input[:,:,start:end]
      for j in range(num_regions):
          #if zcond:
          if i == j:
            messages[k,:,1:,i,j,:] = communication_models[k][i][j](xs_input).numpy() - xs_i
          else:
            msg = communication_models[k][i][j](xs_input).numpy()
            msg0 = 0
            if conditional_msgs:
              msg0 = communication_models[k][i][j](xs_input0s[j]).numpy()
            messages[k,:,1:,i,j,:] = msg - msg0

      if input_regions == 'all' or (isinstance(input_regions, list) and i in input_regions):
        # External input (us) effects, one per dim
        for l in range(num_regions, num_regions+input_len):
          inputs = [xs_input, us_input]
          msg = communication_models[k][i][l](inputs).numpy()
          msg0 = 0
          if conditional_inputs:
            inputs0 = [xs_input, us_input0s[l-num_regions]]
            msg0 = communication_models[k][i][l](inputs0).numpy()
          messages[k, :, 1:, i, l, :] = msg - msg0

  # Zero out padded sequences:
  assert len(trial_lengths) == num_trials
  for i, tl in enumerate(trial_lengths):
    messages[:,i,tl:,:,:,:] = 0

  # Marginalize over z
  Ezs = np.exp(result_dict['z_posterior_ll'].numpy())[:,:,:]
  Ezs = np.moveaxis(Ezs, -1, 0)
  Ems = np.sum(Ezs[:,:,:,None,None,None] * messages, axis=0)

  # Norm
  msg_input_norms = np.linalg.norm(Ems, axis=-1)

  return Ems, msg_input_norms


def get_msgs_and_norms(communication_models, result_dict, num_states, num_regions, latent_dim,
                       latent_region_sizes, trial_lengths, input_len, input_regions='all',
                       conditional_msgs=False, conditional_inputs=False, x0_model=False):
  """
  Note locally conditioned model requires concatenating local and second region xs.
  Also note input models takes in entire x vector across regions and split region internally.
  """
  if x0_model:
    import copy
    communication_models = copy.deepcopy(communication_models)
    tfmap_move = lambda f, x: tf.map_fn(f, np.moveaxis(x, 1, 0))
    for i in range(num_states):
      for j in range(num_regions):
        for k in range(num_regions+input_len):
          communication_models[i,k,j] = tfmap_move(communication_models[i,k,j])
     #tf.map_fn(communication_models[k][i][j], np.moveaxis(xs_input, 1, 0)).numpy()

  xs_input = result_dict['x_sampled'].numpy()[:,:-1,:]
  if len(xs_input.shape) == 4 and xs_input.shape[-2] == 1: # otherwise breaks for d=1 case.
    xs_input = np.squeeze(xs_input,axis=-2)
  us_input = result_dict['us'].numpy()[:,:-1,:]
  num_trials, num_timepoints = xs_input.shape[0:2]

  latent_region_sizes_cs = np.cumsum([0] + latent_region_sizes)
  starts_ends = list(zip(latent_region_sizes_cs[:-1], latent_region_sizes_cs[1:]))

  # Create copies of xs input with each region zeroed out
  if conditional_msgs:
    xs_input0s = []
    for i in range(num_regions):
      start, end = starts_ends[i]
      xs_input0 = np.copy(xs_input)
      xs_input0[:,:,start:end] = 0
      xs_input0s.append(xs_input0)

  if conditional_inputs:
    us_input0s = []
    for l in range(input_len):
      us_input0 = np.copy(us_input)
      us_input0[:,:,l] = 0
      us_input0s.append(us_input0)

  # Get messages
  messages = np.zeros((num_states, num_trials, num_timepoints+1,
                       num_regions, num_regions+input_len, latent_dim))
  messages_diff = np.zeros_like(messages)
  for k in range(num_states):
    for i in range(num_regions):
      start, end = starts_ends[i]
      xs_i = xs_input[:,:,start:end]
      for j in range(num_regions):
          if i == j:
            #if x0_model:
            #  messages[k,:,1:,i,j,:] = tf.map_fn(communication_models[k][i][j], np.moveaxis(xs_input, 1, 0)).numpy() - xs_i
            #else:
            messages[k,:,1:,i,j,:] = communication_models[k][i][j](xs_input).numpy() - xs_i
          else:
            msg = communication_models[k][i][j](xs_input).numpy()
            msg0 = 0
            if conditional_msgs:
              msg0 = communication_models[k][i][j](xs_input0s[j]).numpy()
            messages[k,:,1:,i,j,:] = msg - msg0

      if input_regions == 'all' or (isinstance(input_regions, list) and i in input_regions):
        # External input (us) effects, one per dim
        for l in range(num_regions, num_regions+input_len):
          inputs = [xs_input, us_input]
          msg = communication_models[k][i][l](inputs).numpy()
          msg0 = 0
          if conditional_inputs:
            inputs0 = [xs_input, us_input0s[l-num_regions]]
            msg0 = communication_models[k][i][l](inputs0).numpy()
          messages[k, :, 1:, i, l, :] = msg - msg0

  # Zero out padded sequences:
  assert len(trial_lengths) == num_trials
  for i, tl in enumerate(trial_lengths):
    messages[:,i,tl:,:,:,:] = 0

  # Marginalize over z
  Ezs = np.exp(result_dict['z_posterior_ll'].numpy())[:,:,:]
  Ezs = np.moveaxis(Ezs, -1, 0)
  Ems = np.sum(Ezs[:,:,:,None,None,None] * messages, axis=0)

  # Norm
  msg_input_norms = np.linalg.norm(Ems, axis=-1)

  return Ems, msg_input_norms


def get_grad_field_svae(dynamics_models, x1_range=None, x2_range=None, lr=None,
                        norm=True, eps=1e-4, latent_dims=2, mlen=1):
  # NOTE: should also be able to compute for any dims
  # plotting function should either plot 2d or 3d and project if higher.

  # Consruct x1,x2 grid
  x1_, x2_ = np.meshgrid(x1_range, x2_range)

  all_dvdxs = []
  all_grads = []
  for j in range(len(dynamics_models)):

    all_inputs_ = []
    all_results_ = []
    all_dxs_ = []
    all_fs_ = []
    all_grads_ = []
    for i in range(x1_.shape[0]):

      # Prep grid inputs, zero pad to match dynamics model input shape
      inputs = np.vstack([x1_[i,:], x2_[i,:]])
      if lr is not None:  # Transform grid back to dynamics basis
        inputs_tr = np.linalg.lstsq(lr.coef_, inputs - lr.intercept_[:,None],
                                 rcond=None)[0]
      else:
        inputs_tr = inputs
      inputs_tr = inputs_tr.T

      # Zero pad for second region
      #inputs = np.dstack([inputs, np.zeros_like(inputs)])
      inputs_tensor = tf.convert_to_tensor(inputs_tr, dtype=tf.float32)

      # Dummy inputs NOTE should standardize!
      dummies = np.zeros([x1_.shape[1],2])
      dummies_tensor = tf.convert_to_tensor(dummies, dtype=tf.float32)

      psis_tensor = tf.zeros_like(inputs_tensor)

      # Get model output and grad
      in_ = [inputs_tensor, dummies_tensor, psis_tensor]

      with tf.GradientTape() as g:
        g.watch(in_)
        output = dynamics_models[j](in_) #, xmean=True)
      grad = g.gradient(output, in_)[0].numpy().squeeze()

      result = output.numpy().squeeze()
      # Transform result to the correct basis
      if lr is not None:
        result = lr.predict(result)
        grad = lr.predict(grad)
      dx = result - inputs.T
      if norm:
        dx = dx / np.linalg.norm(dx+eps)
      all_dxs_.append(dx)
      all_grads_.append(grad)

    dvdx1 = np.asarray(all_dxs_)[:,:,0].squeeze()
    dvdx2 = np.asarray(all_dxs_)[:,:,1].squeeze()
    all_dvdxs.append((dvdx1, dvdx2))
    all_grads.append(np.asarray(all_grads_))

  return x1_, x2_, all_dvdxs, all_grads

# NOTE WE HAVE TWO OF THESE FUNCTIONS
# NOTE: need to implement
def get_msgs_and_norms_svae(communication_models, result_dict, num_states, num_regions, latent_dim,
                       latent_region_sizes, trial_lengths, input_len, input_regions='all',
                       conditional_msgs=False, conditional_inputs=False, x0_model=False):
  """
  Note locally conditioned model requires concatenating local and second region xs.
  Also note input models takes in entire x vector across regions and split region internally.
  """
  xs_input = result_dict['x_sampled'].numpy()[:,:-1,:]
  print('xshape', xs_input.shape)
  if len(xs_input.shape) == 4 and xs_input.shape[-2] == 1: # otherwise breaks for d=1 case.
    xs_input = np.squeeze(xs_input,axis=-2)
  print('xshape', xs_input.shape)
  us_input = result_dict['us'].numpy()[:,:-1,:]
  num_trials, num_timepoints = xs_input.shape[0:2]

  latent_region_sizes_cs = np.cumsum([0] + latent_region_sizes)
  starts_ends = list(zip(latent_region_sizes_cs[:-1], latent_region_sizes_cs[1:]))

  # Create copies of xs input with each region zeroed out
  if conditional_msgs:
    xs_input0s = []
    for i in range(num_regions):
      start, end = starts_ends[i]
      xs_input0 = np.copy(xs_input)
      xs_input0[:,:,start:end] = 0
      xs_input0s.append(xs_input0)

  if conditional_inputs:
    us_input0s = []
    for l in range(input_len):
      us_input0 = np.copy(us_input)
      us_input0[:,:,l] = 0
      us_input0s.append(us_input0)

  # For svae need to input dummy psis
  psis = np.zeros_like(xs_input)

  # Get messages
  messages = np.zeros((num_states, num_trials, num_timepoints+1,
                       num_regions, num_regions+input_len, latent_dim))
  messages_diff = np.zeros_like(messages)
  for k in range(num_states):
    for i in range(num_regions):
      start, end = starts_ends[i]
      xs_i = xs_input[:,:,start:end]
      for j in range(num_regions):
          if i == j:
            messages[k,:,1:,i,j,:] = communication_models[k][i][j](xs_input, psis).numpy() - xs_i
          else:
            msg = communication_models[k][i][j](xs_input, psis).numpy()
            msg0 = 0
            if conditional_msgs:
              msg0 = communication_models[k][i][j](xs_input0s[j]).numpy()
            messages[k,:,1:,i,j,:] = msg - msg0

      if input_regions == 'all' or (isinstance(input_regions, list) and i in input_regions):
        # External input (us) effects, one per dim
        for l in range(num_regions, num_regions+input_len):
          inputs = [xs_input, us_input, psis]
          msg = communication_models[k][i][l](inputs).numpy()
          if i < 2 and l == 5:
            print(msg.shape)
            print(k, 'sum msg', i, l, np.sum(msg))
          msg0 = 0
          if conditional_inputs:
            inputs0 = [xs_input, us_input0s[l-num_regions]]
            msg0 = communication_models[k][i][l](inputs0).numpy()
            if i < 2 and l == 5:
              print(k, 'sum msg0', i, l, np.sum(msg0))
          messages[k, :, 1:, i, l, :] = msg - msg0

  # Zero out padded sequences:
  assert len(trial_lengths) == num_trials
  for i, tl in enumerate(trial_lengths):
    messages[:,i,tl:,:,:,:] = 0

  # Marginalize over z
  Ezs = np.exp(result_dict['z_posterior_ll'].numpy()) #[:,:,:]
  Ezs = np.moveaxis(Ezs, -1, 0)
  if num_states > 1:
    Ems = np.sum(Ezs[:,:,:,None,None,None] * messages, axis=0)
  else:
    Ems = np.squeeze(messages, axis=0)

  # Norm
  msg_input_norms = np.linalg.norm(Ems, axis=-1)

  return Ems, msg_input_norms


def get_grad_field_mr(dynamics_models, x1_ranges=None, x2_ranges=None, lrs=None,
                      norm=True, eps=1e-4, latent_dims=[2,2,2], mlen=74,
                      num_uinputs=2, svae=False, task_id=None):
  # NOTE: should also be able to compute for any dims
  # plotting function should either plot 2d or 3d and project if higher.
  num_regions = len(latent_dims)
  latent_dims_cumsum = np.hstack([0, np.cumsum(latent_dims)])

  # Consruct x1,x2 grid
  x1s = []
  x2s = []
  for x1_range, x2_range in zip(x1_ranges, x2_ranges):
    x1_, x2_ = np.meshgrid(x1_range, x2_range)
    x1s.append(x1_)
    x2s.append(x2_)

  all_dvdxs = []
  all_grads = []
  for j in range(len(dynamics_models)):

    print('dynamics ', j)

    # NOTE need to transform per region, but zero pad to pass into net seperately.
    # Since we don't want any communication influence here.
    # Then transform per region after

    region_dvdxs = []
    region_grads = []
    for r in range(num_regions):

      print('region', r)
      s, e = latent_dims_cumsum[r:r+2]

      # The meshgrids could be different sizes per region
      x1_ = x1s[r]
      x2_ = x2s[r]

      all_dxs_ = []
      all_grads_ = []
      for i in range(x1s[r].shape[0]):

        # Prep grid inputs, zero pad to match dynamics model input shape
        inputs = np.vstack([x1_[i,:], x2_[i,:]])
        if lrs is not None:  # Transform grid back to dynamics basis
          inputs_tr = np.linalg.lstsq(lrs[r].coef_, inputs-lrs[r].intercept_[:,None],
                                      rcond=None)[0]
        else:
          inputs_tr = inputs
        zeros = np.zeros([1,mlen-x1_.shape[1],latent_dims[r]])
        inputs_tr = np.hstack([inputs_tr.T[np.newaxis,:,:],zeros])

        # Zero pad for other regions
        inputs_tr_padded = np.zeros([1, mlen, np.sum(latent_dims)])
        inputs_tr_padded[:,:,s:e] = inputs_tr
        inputs_tensor = tf.convert_to_tensor(inputs_tr_padded, dtype=tf.float32)

        # Dummy inputs NOTE should standardize!
        dummies = np.zeros([1,mlen,num_uinputs])
        dummies_tensor = tf.convert_to_tensor(dummies, dtype=tf.float32)

        in_ = [inputs_tensor[:,:,:], dummies_tensor]
        if svae:
          psis_tensor = tf.zeros_like(inputs_tensor)
          in_.append(psis_tensor)
        if task_id is not None:
          task_ids_tensor = tf.zeros((1, mlen, 1)) + task_id
          in_.append(task_ids_tensor)

        # Get model output and grad
        with tf.GradientTape() as g:
          g.watch(in_)
          output = dynamics_models[j](in_)
        grad = g.gradient(output, in_)[0][:,:x1_.shape[1],s:e].numpy().squeeze()
        result = output[:,:x1_.shape[1],s:e].numpy().squeeze()

        # Transform result to the correct basis
        if lrs is not None:
          grad_trans = lrs[r].predict(grad)
          result_trans = lrs[r].predict(result)
        else:
          grad_trans = grad
          result_trans = result

        dx = result_trans - inputs.T
        if norm:
          dx = dx / np.linalg.norm(dx+eps)
        all_dxs_.append(dx)
        all_grads_.append(grad_trans)

      dvdx1 = np.asarray(all_dxs_)[:,:,0].squeeze()
      dvdx2 = np.asarray(all_dxs_)[:,:,1].squeeze()
      region_dvdxs.append((dvdx1, dvdx2))
      region_grads.append(np.asarray(all_grads_))

    all_dvdxs.append(region_dvdxs)
    all_grads.append(region_grads)

  return x1s, x2s, all_dvdxs, all_grads


def get_grad_field_mr_zcond(dynamics_models, x1_ranges=None, x2_ranges=None, lrs=None,
                            norm=True, eps=1e-4, latent_dims=[2,2], mlen=74, num_uinputs=2,
                            task_id=None, svae=False):
  # NOTE: should also be able to compute for any dims
  # plotting function should either plot 2d or 3d and project if higher.
  num_regions = len(latent_dims)
  latent_dims_cumsum = np.hstack([0, np.cumsum([2,2,2])])
  num_states = len(dynamics_models)

  zs_onehot = []
  for i in range(num_states):
    z_onehot = np.zeros([1,1,num_states])
    z_onehot[:,:,i] = 1.0
    zs_onehot.append(tf.convert_to_tensor(z_onehot, dtype_hint=tf.float32))

  # Consruct x1,x2 grid
  x1s = []
  x2s = []
  for x1_range, x2_range in zip(x1_ranges, x2_ranges):
    x1_, x2_ = np.meshgrid(x1_range, x2_range)
    x1s.append(x1_)
    x2s.append(x2_)

  all_dvdxs = []
  all_grads = []
  for j in range(len(dynamics_models)):

    print('dynamics ', j)

    # Discrete state input
    z_onehot = np.zeros([1,1,num_states])
    z_onehot[:,:,j] = 1.0
    zstate_tensor = tf.tile(zs_onehot[j],[1,mlen,1])

    # NOTE need to transform per region, but zero pad to pass into net seperately.
    # Since we don't want any communication influence here.
    # Then transform per region after

    region_dvdxs = []
    region_grads = []
    for r in range(num_regions):

      print('region', r)
      s, e = latent_dims_cumsum[r:r+2]

      # The meshgrids could be different sizes per region
      x1_ = x1s[r]
      x2_ = x2s[r]

      all_dxs_ = []
      all_grads_ = []
      for i in range(x1s[r].shape[0]):

        # Prep grid inputs, zero pad to match dynamics model input shape
        inputs = np.vstack([x1_[i,:], x2_[i,:]])
        if lrs is not None:  # Transform grid back to dynamics basis
          inputs_tr = np.linalg.lstsq(lrs[r].coef_, inputs-lrs[r].intercept_[:,None],
                                      rcond=None)[0]
        else:
          inputs_tr = inputs
        zeros = np.zeros([1,mlen-x1_.shape[1],latent_dims[r]])
        inputs_tr = np.hstack([inputs_tr.T[np.newaxis,:,:],zeros])

        # Zero pad for other regions
        inputs_tr_padded = np.zeros([1, mlen, np.sum(latent_dims)])
        inputs_tr_padded[:,:,s:e] = inputs_tr
        inputs_tensor = tf.convert_to_tensor(inputs_tr_padded, dtype=tf.float32)

        print(inputs_tensor.shape, inputs_tr_padded.shape, inputs_tr.shape, zeros.shape, inputs.shape)

        # Dummy inputs NOTE should standardize!
        dummies = np.zeros([1,mlen,num_uinputs])
        dummies_tensor = tf.convert_to_tensor(dummies, dtype=tf.float32)

        # Get model output and grad
        in_ = [inputs_tensor, dummies_tensor, zstate_tensor]
        if svae:
          psis_tensor = tf.zeros_like(inputs_tensor)
          in_ = [inputs_tensor, dummies_tensor, psis_tensor, zstate_tensor]  # is this right order?
        if task_id is not None:
          task_ids_tensor = tf.zeros((1, mlen, 1)) + task_id
          in_.append(task_ids_tensor)

        print(inputs_tensor.shape, dummies_tensor.shape, zstate_tensor.shape)
        with tf.GradientTape() as g:
          g.watch(in_)
          output = dynamics_models[j](in_)
        grad = g.gradient(output, in_)[0][:,:x1_.shape[1],s:e].numpy().squeeze()
        result = output[:,:x1_.shape[1],s:e].numpy().squeeze()

        # Transform result to the correct basis
        if lrs is not None:
          grad_trans = lrs[r].predict(grad)
          result_trans = lrs[r].predict(result)
        else:
          grad_trans = grad
          result_trans = result

        dx = result_trans - inputs.T
        if norm:
          dx = dx / np.linalg.norm(dx+eps)
        all_dxs_.append(dx)
        all_grads_.append(grad_trans)

      dvdx1 = np.asarray(all_dxs_)[:,:,0].squeeze()
      dvdx2 = np.asarray(all_dxs_)[:,:,1].squeeze()
      region_dvdxs.append((dvdx1, dvdx2))
      region_grads.append(np.asarray(all_grads_))

    all_dvdxs.append(region_dvdxs)
    all_grads.append(region_grads)

  return x1s, x2s, all_dvdxs, all_grads


def generate_1step_trajectories(model, xs_sampled, us, zs_post=None, psis=None):

  print('xs shape gen1', xs_sampled.shape)

  # Squeeze sample dimension
  if xs_sampled.shape[-2] == 1:
    xs_sampled = np.squeeze(xs_sampled, axis=-2)

  # ---- 1 step gen trajectories ----

  xx = xs_sampled[:,:-1,:]
  uu = us[:,1:,:]   # used to be :-1
  inputs = [xx, uu]
  f = model.x_tran
  if psis is not None:
    inputs.append(psis[:,:-1,:])
    f = model.x_tran_time # for svae the regular model take single timestep input
  xpred = f(*inputs).mean().numpy()

  if zs_post is not None:
    # Marginalize over z
    z_weights = np.exp(zs_post[:,:-1,:,np.newaxis])
    xs_gen1 = np.sum(xpred * z_weights, axis=-2)
    xs_gen1 = np.hstack([xx[:,:1,:], xs_gen1])
  else:
    xs_gen1 = np.hstack([xx[:,:1,:], np.squeeze(xpred)])

  # Generate ys from 1 step xs.
  ys_gen1 = model.y_emit(xs_gen1).mean().numpy()
  return xs_gen1, ys_gen1


def generate_trajectories(model, xics, zics, T, latent_dims, obs_dims,
                          us=None, seed=0, sample_x=True, sample_y=False,
                          zs_true=None, zcond=False):

  nt = len(xics)
  multistate = True if model.num_states > 1 else False

  zs = np.zeros((nt, T))
  xs = np.zeros((nt, T, latent_dims))
  ys = np.zeros((nt, T, obs_dims))
  if us is None:
    us = np.zeros((nt, T, 2))

  xs[:,0,:] = xics
  zs[:,0] = np.squeeze(zics)
  for t in range(1,T):

    pad = np.zeros((nt, T-2, latent_dims))
    xpad = np.hstack([xs[:,t-1,:][:,np.newaxis,:], pad])
    pad = np.zeros((nt, T-2, us.shape[-1]))
    upad = np.hstack([us[:,t,:][:,np.newaxis,:], pad])

    # Optionally don't pass us to xtransition handled inside xtran
    # Sample continuous state
    x_dist = model.x_tran(xpad, upad)
    if sample_x:
      xsample = x_dist.sample().numpy()[:,0,:].squeeze()
    else:
      xsample = x_dist.mean().numpy()[:,0,:].squeeze()

    # Sample discrete state
    if multistate:

      if zs_true is not None:
        zs[:,t] = zs_true[:,t].squeeze()
      else:

        # Optionally pass u or x to ztransition
        ztransition_inputs = None
        if model.xz:
          ztransition_inputs = xpad
          if model.uz: # NOTE Input driven switches case
            ztransition_input = tf.concat([ztransition_inputs, upad], axis=-1)
        elif model.uz:
          ztransition_inputs = upad

        # Get unnormalized discrete state transition matrix
        # Note this should really go in the model object. issue with tf?
        if t == 0:
          print('ztran inputs', ztransition_inputs.shape)
        z_sample = model.z_tran(ztransition_inputs)
        log_prob_zt_ztm1 = normalize_logprob(z_sample, axis=-2)[0].numpy()
        prob_zt_ztm1 = np.exp(log_prob_zt_ztm1[:,0,:,:].squeeze())
        prob_zt = []
        for tr in range(nt):
          prob_zt.append(prob_zt_ztm1[tr,:,zs[tr,t-1].astype(int)])
        prob_zt = np.array(prob_zt)
        #if sample_z:
        #  zsample = np.argmax(prob_zt, axis=-1)
        #else:
        zsample = np.argmax(prob_zt, axis=-1)
        zs[:,t] = zsample

      # NOTE should marginalize here, not take argmax.
      # argmax state
      xsample_ = []
      for tr in range(nt):
        xsample_.append(xsample[tr,int(zs[tr,t]),:])
      xsample = np.array(xsample_)

    xs[:,t,:] = xsample

    # Sample emission
    pad = np.zeros((nt, T-1, latent_dims))
    xpad = np.hstack([xs[:,t,:][:,np.newaxis,:], pad])
    if sample_y:
      ysample = tf.squeeze(model.y_emit(xpad).sample()[:,0,:])
    else:
      ysample = tf.squeeze(model.y_emit(xpad).mean()[:,0,:])
    ys[:,t,:] = ysample.numpy()

  return zs, xs, ys


def generate_trajectories_marginalize(model, xics, zics, T, latent_dims, obs_dims,
                          us=None, seed=0, sample_x=False, sample_y=False,
                          zs_true=None, zcond=False):

  nt = len(xics)
  multistate = True if model.num_states > 1 else False

  zs = np.zeros((nt, T, num_states))
  xs = np.zeros((nt, T, latent_dims))
  ys = np.zeros((nt, T, obs_dims))
  if us is None:
    us = np.zeros((nt, T, 2))

  xs[:,0,:] = xics
  for tr in range(nt):
    zs[tr,0,zics[tr]] = 1

  for t in range(1,T):

    pad = np.zeros((nt, T-2, latent_dims))
    xpad = np.hstack([xs[:,t-1,:][:,np.newaxis,:], pad])
    pad = np.zeros((nt, T-2, us.shape[-1]))
    upad = np.hstack([us[:,t,:][:,np.newaxis,:], pad])

    # Sample continuous state
    x_dist = model.x_tran(xpad, upad)
    if sample_x:
      xsample = x_dist.sample().numpy()[:,0,:].squeeze()
    else:
      xsample = x_dist.mean().numpy()[:,0,:].squeeze()

    # Sample discrete state dist and marginalize
    if multistate:

      if zs_true is not None:
        zs[:,t] = zs_true[:,t].squeeze()
      else:

        # Optionally pass u or x to ztransition
        ztransition_inputs = None
        if model.zx:
          ztransition_inputs = xpad
          if model.zu: # NOTE Input driven switches case
            ztransition_input = tf.concat([ztransition_inputs, upad], axis=-1)
        elif model.zu:
          ztransition_inputs = upad

        # Get unnormalized discrete state transition matrix
        # Note this should really go in the model object. issue with tf?
        z_sample = model.z_tran(ztransition_inputs)
        log_prob_zt_ztm1 = normalize_logprob(z_sample, axis=-2)[0].numpy()
        prob_zt_ztm1 = np.exp(log_prob_zt_ztm1[:,0,:,:].squeeze())
        prob_zt = []
        for tr in range(nt):
          prob_zt.append(prob_zt_ztm1[tr,:,zs[tr,t-1].astype(int)])
        prob_zt = np.array(prob_zt)
        #if sample_z:
        #  zsample = np.argmax(prob_zt, axis=-1)
        #else:
        zsample = np.argmax(prob_zt, axis=-1)
        zs[:,t] = zsample

    xs[:,t,:] = xsample

    # Sample emission
    pad = np.zeros((nt, T-1, latent_dims))
    xpad = np.hstack([xs[:,t,:][:,np.newaxis,:], pad])
    if sample_y:
      ysample = tf.squeeze(model.y_emit(xpad).sample()[:,0,:])
    else:
      ysample = tf.squeeze(model.y_emit(xpad).mean()[:,0,:])
    ys[:,t,:] = ysample.numpy()

  return zs, xs, ys


def eval_cosmooth_mse(ys_test, ys_test_dropout_recon,
                      trial_lengths_test, dropout_idxs=None):
  if dropout_idxs is None:
    dropout_idxs = list(np.arange(ys_test.shape[-1]))
  se_test_dropout = []
  for tr, tr_len in enumerate(trial_lengths_test):
    se = np.square(ys_test[tr,:tr_len,dropout_idxs] -
                   ys_test_dropout_recon[tr,:tr_len,dropout_idxs])
    se_test_dropout.append(se)
  mse_test_dropout = np.mean(np.hstack(se_test_dropout))
  return mse_test_dropout


def eval_cosmooth_r2(ys_test, ys_test_dropout_recon,
                     trial_lengths_test, dropout_idxs=None):
  if dropout_idxs is None:
    dropout_idxs = list(np.arange(ys_test.shape[-1]))
  r2_test_dropout = []
  for tr, tr_len in enumerate(trial_lengths_test):
    r2_ = r2_score(ys_test[tr,:tr_len,dropout_idxs],
                   ys_test_dropout_recon[tr,:tr_len,dropout_idxs])
    r2_test_dropout.append(r2_)
  mean_r2_test_dropout = np.mean(np.hstack(r2_test_dropout))
  return mean_r2_test_dropout


def fit_xs(xs_true, xs_inf):
  nt, tl, xdim = xs_true.shape[:]
  lr = LinearRegression(fit_intercept=True)
  lr.fit(xs_inf.reshape(nt*tl,xdim), xs_true.reshape(nt*tl,xdim))
  xs_pred_trans = lr.predict(xs_inf.reshape(nt*tl,xdim)).reshape(nt, tl, xdim)
  return lr, xs_pred_trans


def r2_score(y, ybar):
  min_eps = 1e-9
  ssres = np.sum(np.square(y-ybar))
  sstot = max(np.sum(np.square(y-np.mean(y))), min_eps)
  return 1 - (ssres/sstot)


def per_timepoint_r2(xs_true, xs_gen):
  tl = xs_true.shape[1]
  r2s = []
  for t in range(tl):
    r2s.append(r2_score(xs_true[:,t,:], xs_gen[:,t,:]))
  return r2s


def sliding_window_r2(xs_true, xs_gen, size=None):
  nt, tl, xdim = xs_true.shape
  if size is None:
    size = int(tl/20)
  r2s = []
  for t in range(size, tl):
    xs_true_w = xs_true[:,t-size:t,:].reshape(nt, size*xdim)
    xs_gen_w = xs_gen[:,t-size:t,:].reshape(nt, size*xdim)
    r2s.append(r2_score(xs_true_w, xs_gen_w))
  return r2s


def per_timepoint_mse(xs_true, xs_gen):
  tl = xs_true.shape[1]
  mses = []
  for t in range(tl):
    mses.append(np.mean(np.square(xs_true[:,t,:]-xs_gen[:,t,:])))
  return mses


def transform_latents(lrs, xs_inf, xdim=None):
  nt, tl, xdim_ = xs_inf.shape
  if xdim is None:
    xdim = xdim_
  xs_inf_trans = []
  for i in range(len(lrs)):
    s = (i*xdim)
    xs_inf_flat = xs_inf[:,:,s:s+xdim].reshape(nt*(tl),xdim)
    _ = lrs[i].predict(xs_inf_flat).reshape(nt,tl,xdim)
    xs_inf_trans.append(_)
  xs_inf_trans = np.concatenate(xs_inf_trans, axis=-1)
  return xs_inf_trans


def inv_transform_latents(lrs, ics, xdim=None):
  if xdim is None:
    xdim = ics.shape[-1]
  ics_trans = []
  for i in range(len(lrs)):
    s = (i*xdim)
    bb = np.linalg.lstsq(lrs[i].coef_, (ics[:,s:s+xdim] - lrs[i].intercept_[:,None].T).T, #,None
                         rcond=None)[0].T # Invert the affine map
    ics_trans.append(bb)
  ics_trans = np.concatenate(ics_trans, axis=-1)
  return ics_trans

