
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import multivariate_normal
import tensorflow as tf


def lotka_volterra_params():
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
  return params1, params2


def comp_lotka_volterra_dyn(xt, yt, params):
  alpha, beta, gamma, delta = params
  dx = alpha * xt - beta * xt * yt
  dy = delta * xt * yt - gamma * yt
  return dx, dy


def generate_switching_lotka_volterra_multiswitch2(C_mat, tswitches=None, params=None,
                                                  T=75000, sample=500, latent_dims=2,
                                                  obs_dims=30, dt=0.001,
                                                  x0=np.array([1.0, 1.0]), noise_level=1.5e-4, #1e-3,
                                                  dtype=np.float32, return_latent=True, nl_emissions=False,
                                                  nl_emissions_type='gaussian_bump',
                                                  regime=None, random_seed=None, random_zic=False):

  if random_seed is not None:
    np.random.seed(random_seed)

  if params is None:
    params1, params2 = lotka_volterra_params()
  else:
    params1, params2 = params

  xs = np.zeros(T)
  ys = np.zeros(T)
  xs[0] = x0[0]
  ys[0] = x0[1]

  z_ic = 0
  if random_zic and regime is None:
    z_ic = np.random.binomial(1, 0.5)

  zs = np.zeros(T).astype(int) + z_ic
  if regime is not None:
    zs += regime

  # Set the switches
  tswitches.append(T)
  for i in range(len(tswitches)-1):
    zs[tswitches[i]:tswitches[i+1]] = int(not(zs[tswitches[i]-1]))

  printt = True

  for t in np.arange(1, T):

    xt = xs[t-1]
    yt = ys[t-1]

    if zs[t] == 0:
      dx, dy = comp_lotka_volterra_dyn(xt, yt, params1)
    else:
      if printt:
        print(t)
        printt = False
      dx, dy = comp_lotka_volterra_dyn(xt, yt, params2)
      dx = -1.0 * dx
      dy = -1.0 * dy

    xs[t] = xt + dx * dt * 1
    ys[t] = yt + dy * dt * 1

  xnew = np.array([xs[t] for t in range(0, T, sample)])
  ynew = np.array([ys[t] for t in range(0, T, sample)])
  znew = np.array([zs[t] for t in range(0, T, sample)])

  xfull = np.hstack((xnew[:, None], ynew[:, None]))
  xfull += np.random.normal(scale=noise_level, size=xfull.shape)

  # Map to observations
  yfull = xfull @ C_mat.T + np.random.randn(100)

  # Add nonlinearity
  if nl_emissions:

    if nl_emissions_type == 'gaussian_bump':
      print('gaussian bump')
      nn = int(xfull.shape[-1]/2)
      x1_means = np.linspace(0,2.5,nn)
      x1_vars = [0.1] * nn

      ys_gb1 = np.array([multivariate_normal.pdf(xfull[:,0], mean=x1_means[nidx], cov=x1_vars[nidx]) / 1.5 #* 150
                       for nidx in range(nn)])
      ys_gb2 = np.array([multivariate_normal.pdf(xfull[:,1], mean=x1_means[nidx], cov=x1_vars[nidx]) / 1.5 #* 150
                       for nidx in range(nn)])
      yfull = np.concatenate([ys_gb1, ys_gb2], axis=0).T + np.random.randn(nn*2)

    elif nl_emissions_type == 'from_data':
      raise ValueError('not implemented')

  if return_latent:
    return xfull.astype(dtype), yfull.astype(dtype), znew.astype(int)
  else:
    return yfull.astype(dtype)



def generate_switching_lotka_volterra_multiswitch(C_mat, tswitches=None, params=None,
                                                  T=75000, sample=500, latent_dims=2,
                                                  obs_dims=30, dt=0.001,
                                                  x0=np.array([1.0, 1.0]), noise_level=1.5e-4, #1e-3,
                                                  dtype=np.float32, return_latent=True, nl_emissions=False,
                                                  regime=None, random_seed=None, random_zic=False):

  if random_seed is not None:
    np.random.seed(random_seed)

  if params is None:
    params1, params2 = lotka_volterra_params()
  else:
    params1, params2 = params

  xs = np.zeros(T)
  ys = np.zeros(T)
  xs[0] = x0[0]
  ys[0] = x0[1]

  z_ic = 0
  if random_zic and regime is None:
    z_ic = np.random.binomial(1, 0.5)

  zs = np.zeros(T).astype(int) + z_ic
  if regime is not None:
    zs += regime

  # Set the switches
  tswitches.append(T)
  for i in range(len(tswitches)-1):
    zs[tswitches[i]:tswitches[i+1]] = int(not(zs[tswitches[i]-1]))

  printt = True

  for t in np.arange(1, T):

    xt = xs[t-1]
    yt = ys[t-1]

    if zs[t] == 0:
      dx, dy = comp_lotka_volterra_dyn(xt, yt, params1)
    else:
      if printt:
        print(t)
        printt = False
      dx, dy = comp_lotka_volterra_dyn(xt, yt, params2)
      dx = -1.0 * dx
      dy = -1.0 * dy

    xs[t] = xt + dx * dt * 1
    ys[t] = yt + dy * dt * 1

  xnew = np.array([xs[t] for t in range(0, T, sample)])
  ynew = np.array([ys[t] for t in range(0, T, sample)])
  znew = np.array([zs[t] for t in range(0, T, sample)])

  xfull = np.hstack((xnew[:, None], ynew[:, None]))
  xfull += np.random.normal(scale=noise_level, size=xfull.shape)

  # Map to observations
  yfull = xfull @ C_mat.T + np.random.randn(100)

  # Add nonlinearity
  if nl_emissions:
    #first test, rslds still fits well.
    relu = lambda x: np.maximum(0,x)
    #yfull = relu(yfull)

    C_mat2 = np.random.randn(obs_dims, obs_dims)
    C_mat2[:int(obs_dims), int(obs_dims):] = 0.0
    C_mat2[int(obs_dims):, :int(obs_dims)] = 0.0

    C_mat3 = np.random.randn(obs_dims, obs_dims)
    C_mat3[:int(obs_dims), int(obs_dims):] = 0.0
    C_mat3[int(obs_dims):, :int(obs_dims)] = 0.0

    yfull = relu(relu(yfull) @ C_mat2.T + np.random.randn(100))
    yfull = yfull @ C_mat3.T + np.random.randn(100)

  if return_latent:
    return xfull.astype(dtype), yfull.astype(dtype), znew.astype(int)
  else:
    return yfull.astype(dtype)


def generate_switching_lotka_volterra(C_mat, tswitch=None, params=None,
                                      T=50000, sample=500, latent_dims=2,
                                      obs_dims=30, dt=0.001,
                                      x0=np.array([1.0, 1.0]), noise_level=1.5e-4, #1e-3,
                                      dtype=np.float32, return_latent=True,
                                      regime=None, loc_switch=False, zswitch=False,
                                      random_seed=None, random_zic=False):

  if random_seed is not None:
    np.random.seed(random_seed)

  if params is None:
    params1, params2 = lotka_volterra_params()
  else:
    params1, params2 = params

  xs = np.zeros(T)
  ys = np.zeros(T)
  xs[0] = x0[0]
  ys[0] = x0[1]

  z_ic = 0
  if random_zic and regime is None:
    z_ic = np.random.binomial(1, 0.5)

  zs = np.zeros(T).astype(int) + z_ic
  if regime is not None:
    zs += regime

  # Set the switch
  zs[tswitch:] = int(not(z_ic))

  printt = True

  for t in np.arange(1, T):

    xt = xs[t-1]
    yt = ys[t-1]

    if zs[t] == 0:
      dx, dy = comp_lotka_volterra_dyn(xt, yt, params1)
    else:
      if printt:
        print(t)
        printt = False
      dx, dy = comp_lotka_volterra_dyn(xt, yt, params2)
      dx = -1.0 * dx
      dy = -1.0 * dy

    xs[t] = xt + dx * dt * 1
    ys[t] = yt + dy * dt * 1

  xnew = np.array([xs[t] for t in range(0, T, sample)])
  ynew = np.array([ys[t] for t in range(0, T, sample)])
  znew = np.array([zs[t] for t in range(0, T, sample)])

  xfull = np.hstack((xnew[:, None], ynew[:, None]))
  xfull += np.random.normal(scale=noise_level, size=xfull.shape)

  # Map to observations
  yfull = xfull @ C_mat.T

  if return_latent:
    return xfull.astype(dtype), yfull.astype(dtype), znew.astype(int)
  else:
    return yfull.astype(dtype)


def get_potential_grad(x1_, x2_, params1=None, params2=None, regime=None):

  if params1 is None:
    params = lotka_volterra_params()
  else:
    params = [params1]
    if params2 is not None:
      params.append(params2)

  # NOTE function not done, need to add Vs.
  dVdxs = []
  for k in range(len(params)):
    dvdx1 = np.zeros_like(x1_)
    dvdx2 = np.zeros_like(x1_)
    for i in range(x1_.shape[0]):
      for j in range(x1_.shape[1]):
        d1, d2 = comp_lotka_volterra_dyn(x1_[i,j], x2_[i,j], params[k])
        if k == 1:
          d1 *= -1.0
          d2 *= -1.0
        dvdx1[i,j] = d1
        dvdx2[i,j] = d2
    dVdxs.append([dvdx1, dvdx2])

  if len(dVdxs) == 1:
    dVdxs.append([0,0])

  V_1 = V_2 = 0
  return V_1, *dVdxs[0], V_2, *dVdxs[1]


def get_grad(params1, params2=None):

  all_dvdxs = []
  for pidx, pars in enumerate([params1, params2]):

    x1_ = np.arange(0, 2.5, 0.125)
    x2_ = np.arange(0, 2.0, 0.125)    #.25
    x1_, x2_ = np.meshgrid(x1_, x2_)

    dvdx1 = np.zeros_like(x1_)
    dvdx2 = np.zeros_like(x1_)
    for i in range(x1_.shape[0]):
      for j in range(x1_.shape[1]):
        d1, d2 = comp_lotka_volterra_dyn(x1_[i,j], x2_[i,j], pars)
        if pidx == 1:
          d1 *= -1.0
          d2 *= -1.0
        dvdx1[i,j] = d1
        dvdx2[i,j] = d2
    all_dvdxs.append([dvdx1, dvdx2])

  return all_dvdxs
