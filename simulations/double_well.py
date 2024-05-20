from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy.stats import multivariate_normal

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")


def evidence_process(num_timepoints, means, evidence_period, random_seed=0):
  """tbd."""
  np.random.seed(random_seed)

  # Draw total num of clicks/towers for each side
  clicks = [np.random.poisson(means[i]) for i in range(2)]

  # Sample locations uniformly over timepoints
  locs = [np.random.choice(list(range(*evidence_period)), clicks[i])
          for i in range(2)]

  # Left clicks positive right clicks negative
  inputs = np.zeros((num_timepoints,2))
  #inputs[locs[0],0] += -0.55
  #inputs[locs[1],0] += 0.55
  for loc in locs[0]:
    inputs[loc,0] += -0.55
  for loc in locs[1]:
    inputs[loc,0] += 0.55
  return inputs


def gen_trial_inputs(num_trials, T, means=[[15,30], [30,15]],
                     evidence_period=[3,40], ratio=0.5, random_seed=0, side=None):
  """
  Generate trial inputs. Can also have harder trials [20,10]
  """
  np.random.seed(random_seed)
  us = np.zeros((num_trials, T, 2))
  trial_types = []
  for i in range(num_trials):
    # Pick side
    if side == 'left':
      trial_mean = means[1]
    if side == 'right':
      trial_mean = means[0]
    elif np.random.uniform() > ratio:
      trial_mean = means[0]
    else:
      trial_mean = means[1]
    trial_type = -1 if np.argmax(trial_mean)==0 else 1
    seed = random_seed + i + (num_trials*random_seed)  # added num trials to change across sequences
    us[i,:,:] = evidence_process(T, trial_mean, evidence_period, random_seed=seed)
    trial_types.append(trial_type)
  return us, np.array(trial_types)


def simulate_xs(us, bias, T=70):
  """TBD"""

  # Latent noise
  sigma1 = lambda: np.random.normal(0, 0.0005)
  sigma2 = lambda: np.random.normal(0, 0.1) * np.array([1, 0]) # only on 1st dim

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  def dxdt_accum(x,t,u, bias):
    if x[0] < 0:
      return (np.hstack([x[0] - np.power(x[0],3), -x[1]]) + inputs(u,t) + sigma1())
    else:
      return (np.hstack([ bias*(x[0] - np.power(x[0],3)), -x[1]]) + inputs(u,t) + sigma1())

  # Return dynamics gradient
  a = 0.5
  offset = np.array([0, 0.5])
  dxdt_return = lambda x,t,u: (np.hstack([-a*x[0]+a*offset[0], -a*x[1]+a*offset[1]]) +
                               inputs(u,t) + sigma2())

  dt1 = 0.1
  dt2 = 0.4
  ts = np.arange(0,T/10,dt1)[1:]
  x0 = np.array([0, 0.5])

  # Simulate trials
  x0 = np.array([0, 0.5])
  xs_trials = []
  for u in us:
    x = x0
    xs = [x]
    for i, t in enumerate(ts):
      if i <= 50:
        dxdt_ = dxdt_accum
        dx = dt1*dxdt_(x,t,u, bias)
      else:
        dxdt_ = dxdt_return
        dx = dt2*dxdt_(x,t,u)
      x = x + dx
      xs.append(x)
    xs = np.stack(xs).T
    xs_trials.append(xs)

  return xs_trials, dt1, dt2


def simulate_xs_sr(us, T=70, feedback=False, r1s1_noise=0.008, r1s2_noise=0.035, #0.1,
                            r2_noise=0.1, r1_scale=0.6, r2_scale=4.5*np.array([0.4, 0.7]),
                            history_effect=True):

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise)  # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise*2) * np.array([1, 0.1]) # only on 1st dim, return
  sigma = lambda: np.random.normal(0, r2_noise)  # region2
  A1 = np.array([[0.5, 0.0],[0.0, 0.5]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: 0.1 * (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                                    inputs(u,t) + sigma1())

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  dxdt_return = lambda x,t,u: 1.4 * (r1_scale*(-x + offset) + sigma2())

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])

  r1_state = 0

  # Simulate trials
  xs_trials1 = []
  msgs_trials = []
  switch_idxs = []
  for u in us:

    x1_ = x01
    xs1 = [x1_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r1_state == 0:
        dx1_ = dt1 * dxdt_accum(x1_,t,u)
        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][2].append(inputs(u,t))
      else:
        dx1_ = dt1 * dxdt_return(x1_,t,u)
        msgs[0][0].append(dx1_)
        msgs[0][2].append(np.zeros(2))
      msgs[0][1].append(np.zeros(2))
      msgs[1][0].append(np.zeros(2))
      msgs[1][1].append(np.zeros(2))
      msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_
      xs1.append(x1_)

      if r1_state == 0 and i > 40: #(i > 50 or x1_ >) : #np.any(x2_ >= 2): #t > 50: #
        if np.random.rand() >= 0.5 or i > 50:
          if not accum_only:
            r1_state = 1
            switch_idxs.append(i+1)

    xs1 = np.stack(xs1) #.T
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    x01 = x1_ if history_effect else np.array([0.0, 0.5])

    # Reset r2 state
    r1_state = 0

  return xs_trials1, dt1, msgs_trials, switch_idxs


def powspace(start, stop, power, num):
  start = np.power(start, 1/float(power))
  stop = np.power(stop, 1/float(power))
  return np.power( np.linspace(start, stop, num=num), power)


def simulate_xs_sr2_smooth2(us, T=70, feedback=False,
                           r1s1_noise=0.008, r1s2_noise=0.035,
                           r2_noise=0.1, r1_scale=0.6,
                           r2_scale=4.5*np.array([0.4, 0.7]),
                           history_effect=True, a=-1, accum_only=False, seed=0):

  np.random.seed(seed)

  ps = powspace(0, 1, 1, 100)
  ls = np.linspace(0, 0.5, 100)

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise) * np.array([1, 0.5]) # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise*5) * np.array([7.5, 2]) # only on 1st dim, return  #5.5, 1.5
  sigma = lambda: np.random.normal(0, r2_noise)  # region2
  A1 = np.array([[0.5, 0.0],[0.0, 0.5]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                              inputs(u,t) + sigma1())

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  c = 1 # 0.3

  def dxdt_return(x,t,u):
    b2 = ps[np.abs(ls-x[1]).argmin()]
    b1 = 1 - b2
    nn = 0
    #if x[0] < 0:
    #  nn = np.random.normal(0, 0.5) * np.array([1,0])
    return (b1*np.hstack([-1*x[0], 0.9*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
            b2*np.hstack([-2*x[0], -2*(x[1]-0.5)]) * c + inputs(u,t) + sigma2() + nn)

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])

  r1_state = 0
  xmax = 1

  # Simulate trials
  xs_trials1 = []
  msgs_trials = []
  switch_idxs = [] #0
  for u in us:

    x1_ = x01
    xs1 = [x1_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r1_state == 0:

        dxdt_1 = dxdt_accum
        dx1_ = 0.1 * dt1 * dxdt_1(x1_,t,u)

        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][2].append(inputs(u,t))

      else:

        dxdt_ = dxdt_return
        dxdt_t = dxdt_(x1_,t,u)
        dx1_ = 0.1 * dt1 * dxdt_t # + r2_scale * x2_[1]

        msgs[0][0].append(dxdt_t)
        msgs[0][2].append(np.zeros(2))

      msgs[0][1].append(np.zeros(2))
      msgs[1][0].append(np.zeros(2))
      msgs[1][1].append(np.zeros(2))
      msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_
      xs1.append(x1_)

      if r1_state == 0 and not accum_only:
        if np.abs(x1_[0]) > xmax:
          r1_state = 1
          switch_idxs.append(i+1)

    xs1 = np.stack(xs1) #.T
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    x01 = x1_ if history_effect else np.array([0.0, 0.5])

    # Reset r2 state
    r1_state = 0

  return xs_trials1, dt1, msgs_trials, switch_idxs


def simulate_xs_sr2_smooth(us, T=70, feedback=False,
                           r1s1_noise=0.008, r1s2_noise=0.035,
                           r2_noise=0.1, r1_scale=0.6,
                           r2_scale=4.5*np.array([0.4, 0.7]),
                           history_effect=True, a=-1, accum_only=False):

  ps = powspace(0, 1, 1, 100)
  ls = np.linspace(0, 0.5, 100)

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise) * np.array([1, 0.5]) # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise*5) * np.array([7.5, 2]) # only on 1st dim, return  #5.5, 1.5
  sigma = lambda: np.random.normal(0, r2_noise)  # region2
  A1 = np.array([[0.5, 0.0],[0.0, 0.5]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                              inputs(u,t) + sigma1())

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  c = 1 # 0.3

  def dxdt_return(x,t,u):
    b2 = ps[np.abs(ls-x[1]).argmin()]
    b1 = 1 - b2
    nn = 0
    #if x[0] < 0:
    #  nn = np.random.normal(0, 0.5) * np.array([1,0])
    return (b1*np.hstack([-1*x[0], 0.9*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
            b2*np.hstack([-2*x[0], -2*(x[1]-0.5)]) * c + inputs(u,t) + sigma2() + nn)

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])

  r1_state = 0

  # Simulate trials
  xs_trials1 = []
  msgs_trials = []
  switch_idxs = [] #0
  for u in us:

    x1_ = x01
    xs1 = [x1_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r1_state == 0:

        dxdt_1 = dxdt_accum
        dx1_ = 0.1 * dt1 * dxdt_1(x1_,t,u)

        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][2].append(inputs(u,t))

      else:

        dxdt_ = dxdt_return
        dxdt_t = dxdt_(x1_,t,u)
        dx1_ = 0.1 * dt1 * dxdt_t # + r2_scale * x2_[1]

        msgs[0][0].append(dxdt_t)
        msgs[0][2].append(np.zeros(2))

      msgs[0][1].append(np.zeros(2))
      msgs[1][0].append(np.zeros(2))
      msgs[1][1].append(np.zeros(2))
      msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_
      xs1.append(x1_)

      if r1_state == 0 and i > 40: #(i > 50 or x1_ >) : #np.any(x2_ >= 2): #t > 50: #
        if np.random.rand() >= 0.5 or i > 50:
          if not accum_only:
            r1_state = 1
            switch_idxs.append(i+1)

    xs1 = np.stack(xs1) #.T
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    x01 = x1_ if history_effect else np.array([0.0, 0.5])

    # Reset r2 state
    r1_state = 0

  return xs_trials1, dt1, msgs_trials, switch_idxs


def simulate_xs_mr2_smoothnew(us, T=70, feedback=False,
                              r1s1_noise=0.008, r1s2_noise=0.035,
                              r2_noise=0.1, r1_scale=0.6,
                              r2_scale=4.5*np.array([0.4, 0.7]),
                              history_effect=True, a=-1, accum_only=False):

  ps = powspace(0, 1, 1, 100)
  ls = np.linspace(0, 0.5, 100)

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise) * np.array([1, 0.5]) # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise*5) * np.array([7.5, 2]) # only on 1st dim, return  #5.5, 1.5
  #sigma = lambda: np.random.normal(0, r2_noise)  # region2
  sigma = lambda: np.random.normal(0, r2_noise*2) * np.array([0.05, 0.75]) # accum region2
  sigma_ret = lambda: np.random.normal(0, r2_noise*2) * np.array([0.05, 0.5]) # accum region2

  r2_fp1 = np.array([0.725, 0])
  r2_fp2 = np.array([-0.725, 0])
  switch_eps = 0.05

  #A1 = np.array([[0.5, 0.0],
  #               [0.0, 0.5]])
  A2 = np.array([[0.0, 0.5],
                 [0.0, 0.0]])

  A3 = np.array([[0.0, 0.0],
                 [0.5, 0.0]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                              inputs(u,t) + sigma1())
  #dxdt_accum2 = lambda x,t,u: np.gradient(x[0]*np.exp(-x[0]**2 - x[1]**2)) + sigma()
  dxdt_accum2 = lambda x,t,u: (np.hstack([(1-2*x[0]**2) *  np.exp(-x[0]**2 - x[1]**2),
                                          (-2*x[0]*x[1]) * np.exp(-x[0]**2 - x[1]**2)]) + sigma() + np.array([0.015, 0]))

  dxdt_return2 = lambda x,t,u: - (np.hstack([(1-2*x[0]**2) *  np.exp(-x[0]**2 - x[1]**2),
                                                 (-2*x[0]*x[1]) * np.exp(-x[0]**2 - x[1]**2)]) + sigma_ret() + np.array([0.095,0])) #- np.array([-0.25,0])

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  c = 1 # 0.3

  def dxdt_return(x,t,u):
    b2 = ps[np.abs(ls-x[1]).argmin()]
    b1 = 1 - b2
    nn = 0
    #if x[0] < 0:
    #  nn = np.random.normal(0, 0.5) * np.array([1,0])

    #return (b1*np.hstack([-1*x[0], 0.9*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
    #        b2*np.hstack([-2*x[0], -2*(x[1]-0.5)]) * c + inputs(u,t) + sigma2() + nn)

    return (b1*np.hstack([-1*x[0], 0.55*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
            b2*np.hstack([-2*x[0], -1.*(x[1]-0.45)]) * c + inputs(u,t) + sigma2() + nn)
  #dxdt_return2 = lambda x,t,u: -dxdt_accum2(x,t,u)

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])
  x02 = np.copy(r2_fp2)  # NOTE when no history!

  r1_state = 0
  r2_state = 0

  # Simulate trials
  xs_trials1 = []
  xs_trials2 = []
  msgs_trials = []
  switch_idxs = [] #0
  for u in us:

    x1_ = np.copy(x01)
    x2_ = np.copy(x02)
    xs1 = [x1_]
    xs2 = [x2_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r2_state == 0:

        dxdt_1 = dxdt_accum
        dx1_ = 0.1 * dt1 * dxdt_1(x1_,t,u)
        dx1_2 = np.zeros(2)

        dxdt_2 = dxdt_accum2
        dx2_ = 0.1 * dt1 * dxdt_2(x2_,t,u)
        dx2_1 = 0.1 * dt1 * A2 @ x1_

        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][1].append(dx1_2)
        msgs[0][2].append(inputs(u,t))
        msgs[1][0].append(dx2_1)
        msgs[1][1].append(dx2_)
        msgs[1][2].append(np.zeros(2))

      else:

        dxdt_2 = dxdt_return2
        dx2_ = 0.1 * dt1 * dxdt_2(x2_,t,u)
        dx2_1 = np.zeros(2)

        dxdt_ = dxdt_return
        dxdt_t = dxdt_(x1_,t,u)
        dx1_ = 0.1 * dt1 * dxdt_t # NOTE no feedback for now! + r2_scale * x2_[1]
        dx1_2 = 0.1 * dt1 * A3 @ np.abs(dx2_)

        msgs[0][0].append(dx1_)
        msgs[0][1].append(dx1_2)
        msgs[0][2].append(np.zeros(2))
        msgs[1][0].append(dx2_1)
        msgs[1][1].append(dx2_)
        msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_ + dx1_2
      xs1.append(x1_)
      x2_ = x2_ + dx2_ + dx2_1
      xs2.append(x2_)

      if r2_state == 0 and np.linalg.norm(x2_-r2_fp1) < switch_eps:
        if not accum_only:
          r2_state = 1
          r1_state = 1
          switch_idxs.append(i+1)

    # If no switch happened
    if r2_state == 0:
      switch_idxs.append(i)

    xs1 = np.stack(xs1) #.T
    xs2 = np.stack(xs2) #.T
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    xs_trials2.append(xs2)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    x01 = x1_ if history_effect else x01
    x02 = x2_ if history_effect else x02

    # Reset r2 state
    r1_state = 0
    r2_state = 0

  return xs_trials1, xs_trials2, dt1, msgs_trials, switch_idxs


def simulate_xs_mr2_smoothnew2b(us, T=70, feedback=False,
                              r1s1_noise=0.008, r1s2_noise=0.035,
                              r2_noise=0.1, r1_scale=0.6,
                              r2_scale=4.5*np.array([0.4, 0.7]),
                              history_effect=True, a=-1, accum_only=False,
                              random_seed=0):

  # set seed till 9/25 this wasn't set!
  np.random.seed(random_seed)

  ps = powspace(0, 1, 1, 100)
  ls = np.linspace(0, 0.5, 100)

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise) * np.array([0.05, 0.05]) # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise*5) * np.array([7.5, 2]) # only on 1st dim, return  #5.5, 1.5
  #sigma = lambda: np.random.normal(0, r2_noise)  # region2
  sigma = lambda: np.random.normal(0, r2_noise*2) * np.array([0.025, 0.45]) # accum region2
  sigma_ret = lambda: np.random.normal(0, r2_noise*2) * np.array([0.075, 0.2]) # accum region2

  r2_fp1 = np.array([0.725, 0])
  r2_fp2 = np.array([-0.725, 0])
  switch_eps = 0.095 #9 #75 #15
  print('switch eps', switch_eps)

  #A1 = np.array([[0.5, 0.0],
  #               [0.0, 0.5]])
  A2 = np.array([[0.0, 0.75],
                 [0.0, 0.0]])

  A3 = np.array([[0.0, 0.0],
                 [0.35, 0.0]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                              inputs(u,t) + sigma1())
  #dxdt_accum2 = lambda x,t,u: np.gradient(x[0]*np.exp(-x[0]**2 - x[1]**2)) + sigma()
  dxdt_accum2 = lambda x,t,u: (np.hstack([(1-2*x[0]**2) *  np.exp(-x[0]**2 - x[1]**2),
                                          (-2*x[0]*x[1]) * np.exp(-x[0]**2 - x[1]**2)]) + sigma())

  dxdt_return2 = lambda x,t,u: - (np.hstack([(1-2*x[0]**2) * np.exp(-x[0]**2 - x[1]**2),
                                             (-2*x[0]*x[1]) * np.exp(-x[0]**2 - x[1]**2)]) + sigma_ret()) #- np.array([-0.25,0])

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  c = 1 # 0.3

  def dxdt_return(x,t,u):
    b2 = ps[np.abs(ls-x[1]).argmin()]
    b1 = 1 - b2
    nn = 0
    #if x[0] < 0:
    #  nn = np.random.normal(0, 0.5) * np.array([1,0])

    #return (b1*np.hstack([-1*x[0], 0.9*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
    #        b2*np.hstack([-2*x[0], -2*(x[1]-0.5)]) * c + inputs(u,t) + sigma2() + nn)

    return (b1*np.hstack([-1*x[0], 0.55*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
            b2*np.hstack([-2*x[0], -1.*(x[1]-0.45)]) * c + inputs(u,t) + sigma2() + nn)
  #dxdt_return2 = lambda x,t,u: -dxdt_accum2(x,t,u)

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])
  x02 = np.copy(r2_fp2)  # NOTE when no history!

  r1_state = 0
  r2_state = 0

  # Simulate trials
  xs_trials1 = []
  xs_trials2 = []
  msgs_trials = []
  switch_idxs = [] #0
  for u in us:

    x1_ = np.copy(x01)
    x2_ = np.copy(x02)
    xs1 = [x1_]
    xs2 = [x2_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r2_state == 0:

        dxdt_1 = dxdt_accum
        dx1_ = 0.1 * dt1 * dxdt_1(x1_,t,u) * 0.75
        dx1_2 = np.zeros(2)

        dxdt_2 = dxdt_accum2
        dx2_ = 0.1 * dt1 * 0.8 * dxdt_2(x2_,t,u)  # Using 0.8 to scale down self effect.
        dx2_1 = 0.1 * dt1 * A2 @ x1_ * 1.25 # Scaled A2 up from 0.5 to 0.75

        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][1].append(dx1_2)
        msgs[0][2].append(inputs(u,t))
        msgs[1][0].append(dx2_1)
        msgs[1][1].append(dx2_)
        msgs[1][2].append(np.zeros(2))

      else:

        dxdt_2 = dxdt_return2
        dx2_ = 0.1 * dt1 * dxdt_2(x2_,t,u) + np.array([-0.015,0])
        dx2_1 = np.zeros(2)

        dxdt_ = dxdt_return
        dxdt_t = dxdt_(x1_,t,u)
        # As x2 gets closer to return fp (ic), it pushes x2 less.
        dx1_ = 0.1 * dt1 * 0.6 * dxdt_t # Using 0.8 to scale down self effect
        dx1_2 = 0.025 * dt1 * A3 @ np.abs(dx2_-r2_fp2) # Scaled up A3 from 0.3 to 0.5, added diff here.

        msgs[0][0].append(dx1_)
        msgs[0][1].append(dx1_2)
        msgs[0][2].append(np.zeros(2))
        msgs[1][0].append(dx2_1)
        msgs[1][1].append(dx2_)
        msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_ + dx1_2
      xs1.append(x1_)
      x2_ = x2_ + dx2_ + dx2_1
      xs2.append(x2_)

      #if r2_state == 0 and np.linalg.norm(x2_-r2_fp1) < switch_eps:
      #  if not accum_only:
      #    r2_state = 1
      #    r1_state = 1
      #    switch_idxs.append(i+1)

      # TODO fill in
      if r2_state == 0 and not accum_only:
        #diff = min(1, np.linalg.norm(x2_-r2_fp1))
        diff = min(1, np.abs(x2_[0]-r2_fp1[0]))
        #pswitch = (1 - diff) / 10
        pswitch = np.exp(-40*diff)
        if int(np.random.binomial(1, pswitch)):
          r2_state = 1
          r1_state = 1
          switch_idxs.append(i+1)

    # If no switch happened
    if r2_state == 0:
      switch_idxs.append(i)

    xs1 = np.stack(xs1) #.T
    xs2 = np.stack(xs2) #.T
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    xs_trials2.append(xs2)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    x01 = x1_ if history_effect else x01
    x02 = x2_ if history_effect else x02

    # Reset r2 state
    r1_state = 0
    r2_state = 0

  return xs_trials1, xs_trials2, dt1, msgs_trials, switch_idxs


def simulate_xs_mr2_smoothnew2(us, T=70, feedback=False,
                              r1s1_noise=0.008, r1s2_noise=0.035,
                              r2_noise=0.1, r1_scale=0.6,
                              r2_scale=4.5*np.array([0.4, 0.7]),
                              history_effect=True, a=-1, accum_only=False,
                              random_seed=0):

  # set seed till 9/25 this wasn't set!
  np.random.seed(random_seed)

  ps = powspace(0, 1, 1, 100)
  ls = np.linspace(0, 0.5, 100)

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise) * np.array([1, 0.5]) # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise*5) * np.array([7.5, 2]) # only on 1st dim, return  #5.5, 1.5
  #sigma = lambda: np.random.normal(0, r2_noise)  # region2
  sigma = lambda: np.random.normal(0, r2_noise*2) * np.array([0.05, 2]) # accum region2
  sigma_ret = lambda: np.random.normal(0, r2_noise*2) * np.array([0.1, 0.35]) # accum region2

  r2_fp1 = np.array([0.725, 0])
  r2_fp2 = np.array([-0.725, 0])
  switch_eps = 0.095

  #A1 = np.array([[0.5, 0.0],
  #               [0.0, 0.5]])
  A2 = np.array([[0.0, 0.75],
                 [0.0, 0.0]])

  A3 = np.array([[0.0, 0.0],
                 [0.75, 0.0]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                              inputs(u,t) + sigma1())
  #dxdt_accum2 = lambda x,t,u: np.gradient(x[0]*np.exp(-x[0]**2 - x[1]**2)) + sigma()
  dxdt_accum2 = lambda x,t,u: (np.hstack([(1-2*x[0]**2) *  np.exp(-x[0]**2 - x[1]**2),
                                          (-2*x[0]*x[1]) * np.exp(-x[0]**2 - x[1]**2)]) + sigma())

  dxdt_return2 = lambda x,t,u: - (np.hstack([(1-2*x[0]**2) * np.exp(-x[0]**2 - x[1]**2),
                                             (-2*x[0]*x[1]) * np.exp(-x[0]**2 - x[1]**2)]) + sigma_ret()) #- np.array([-0.25,0])

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  c = 1 # 0.3

  def dxdt_return(x,t,u):
    b2 = ps[np.abs(ls-x[1]).argmin()]
    b1 = 1 - b2
    nn = 0
    #if x[0] < 0:
    #  nn = np.random.normal(0, 0.5) * np.array([1,0])

    #return (b1*np.hstack([-1*x[0], 0.9*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
    #        b2*np.hstack([-2*x[0], -2*(x[1]-0.5)]) * c + inputs(u,t) + sigma2() + nn)

    return (b1*np.hstack([-1*x[0], 0.55*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
            b2*np.hstack([-2*x[0], -1.*(x[1]-0.45)]) * c + inputs(u,t) + sigma2() + nn)
  #dxdt_return2 = lambda x,t,u: -dxdt_accum2(x,t,u)

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])
  x02 = np.copy(r2_fp2)  # NOTE when no history!

  r1_state = 0
  r2_state = 0

  # Simulate trials
  xs_trials1 = []
  xs_trials2 = []
  msgs_trials = []
  switch_idxs = [] #0
  for u in us:

    x1_ = np.copy(x01)
    x2_ = np.copy(x02)
    xs1 = [x1_]
    xs2 = [x2_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r2_state == 0:

        dxdt_1 = dxdt_accum
        dx1_ = 0.1 * dt1 * dxdt_1(x1_,t,u)
        dx1_2 = np.zeros(2)

        dxdt_2 = dxdt_accum2
        dx2_ = 0.1 * dt1 * 0.8 * dxdt_2(x2_,t,u)  # Using 0.8 to scale down self effect.
        dx2_1 = 0.1 * dt1 * A2 @ x1_  # Scaled A2 up from 0.5 to 0.75

        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][1].append(dx1_2)
        msgs[0][2].append(inputs(u,t))
        msgs[1][0].append(dx2_1)
        msgs[1][1].append(dx2_)
        msgs[1][2].append(np.zeros(2))

      else:

        dxdt_2 = dxdt_return2
        dx2_ = 0.1 * dt1 * dxdt_2(x2_,t,u)
        dx2_1 = np.zeros(2)

        dxdt_ = dxdt_return
        dxdt_t = dxdt_(x1_,t,u)
        # As x2 gets closer to return fp (ic), it pushes x2 less.
        dx1_ = 0.1 * dt1 * 0.8 * dxdt_t # Using 0.8 to scale down self effect
        dx1_2 = 0.1 * dt1 * A3 @ np.abs(dx2_-r2_fp2) # Scaled up A3 from 0.3 to 0.5, added diff here.

        msgs[0][0].append(dx1_)
        msgs[0][1].append(dx1_2)
        msgs[0][2].append(np.zeros(2))
        msgs[1][0].append(dx2_1)
        msgs[1][1].append(dx2_)
        msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_ + dx1_2
      xs1.append(x1_)
      x2_ = x2_ + dx2_ + dx2_1
      xs2.append(x2_)

      if r2_state == 0 and np.linalg.norm(x2_-r2_fp1) < switch_eps:
        if not accum_only:
          r2_state = 1
          r1_state = 1
          switch_idxs.append(i+1)

    # If no switch happened
    if r2_state == 0:
      switch_idxs.append(i)

    xs1 = np.stack(xs1) #.T
    xs2 = np.stack(xs2) #.T
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    xs_trials2.append(xs2)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    x01 = x1_ if history_effect else x01
    x02 = x2_ if history_effect else x02

    # Reset r2 state
    r1_state = 0
    r2_state = 0

  return xs_trials1, xs_trials2, dt1, msgs_trials, switch_idxs


def simulate_xs_mr2_smooth(us, T=70, feedback=False, r1s1_noise=0.008,
                                    r1s2_noise=0.035, r2_noise=0.1, r1_scale=0.6,
                                    r2_scale=4.5*np.array([0.4, 0.7]),
                                    history_effect=True, a=-1, accum_only=False):

  ps = powspace(0, 1, 1, 100)
  ls = np.linspace(0, 0.5, 100)

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise)  # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise*5) * np.array([7.5, 2]) # only on 1st dim, return  #5.5, 1.5
  sigma = lambda: np.random.normal(0, r2_noise)  # region2
  A1 = np.array([[0.5, 0.0],[0.0, 0.5]])
  A2 = np.array([[0.1, 0.0],[0.0, 0.1]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                              inputs(u,t) + sigma1())

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  c = 1 # 0.3

  def dxdt_return(x,t,u):
    b2 = ps[np.abs(ls-x[1]).argmin()]
    b1 = 1 - b2
    nn = 0
    #if x[0] < 0:
    #  nn = np.random.normal(0, 0.5) * np.array([1,0])
    return (b1*np.hstack([-1*x[0], 0.9*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
            b2*np.hstack([-2*x[0], -2*(x[1]-0.5)]) * c + inputs(u,t) + sigma2() + nn)

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])
  x02 = np.array([0.0, 0.0])

  r1_state = 0
  r2_state = 0

  # Simulate trials
  xs_trials1 = []
  xs_trials2 = []
  msgs_trials = []
  switch_idxs = []
  for u in us:

    x1_ = x01
    xs1 = [x1_]
    x2_ = x02
    xs2 = [x2_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r2_state == 0:

        dxdt_1 = dxdt_accum
        dx1_ = 0.1 * dt1 * dxdt_1(x1_,t,u)
        x2_ = A1 @ x2_ + (1-2*x1_[1]) + s

        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][1].append(np.zeros(2))
        msgs[0][2].append(inputs(u,t))

      else:

        dxdt_ = dxdt_return
        dxdt_t = dxdt_(x1_,t,u)
        dx1_ = 0.1 * dt1 * dxdt_t # + r2_scale * x2_[1]
        s = np.array([sigma(), sigma()])
        x2_ = A2 @ x2_ + (1-2*x1_[1]) + s

        msgs[0][0].append(dxdt_t)
        msgs[0][1].append(r2_scale * x2_[1])
        msgs[0][2].append(np.zeros(2))

      msgs[1][0].append(np.array([1-2*x1_[1]] * 2))
      msgs[1][1].append(A2@x2_-x2_ + s)

      # Region 2 always gets no inputs
      msgs[1][2].append(np.zeros(2))

      msgs[0][1].append(np.zeros(2))
      msgs[1][0].append(np.zeros(2))
      msgs[1][1].append(np.zeros(2))
      msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_
      xs1.append(x1_)
      xs2.append(x2_)

      if r2_state == 0 and i > 40: #(i > 50 or x1_ >) : #np.any(x2_ >= 2): #t > 50: #
        if np.random.rand() >= 0.5 or i > 50:
          if not accum_only:
            r2_state = 1
            switch_idxs.append(i+1)

    xs1 = np.stack(xs1) #.T
    xs2 = np.stack(xs2)
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    xs_trials2.append(xs2)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    if history_effect:
      x01 = x1_
      x02 = x2_

    # Reset r2 state
    r2_state = 0

  return xs_trials1, xs_trials2, dt1, msgs_trials, switch_idxs


def simulate_xs_sr2(us, T=70, feedback=False, r1s1_noise=0.008,
                    r1s2_noise=0.035, r2_noise=0.1, r1_scale=0.6,
                    r2_scale=4.5*np.array([0.4, 0.7]),
                    history_effect=True, a=-1, accum_only=False):

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise)  # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise*5) * np.array([7.5, 2]) # only on 1st dim, return  #5.5, 1.5
  sigma = lambda: np.random.normal(0, r2_noise)  # region2
  A1 = np.array([[0.5, 0.0],[0.0, 0.5]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                              inputs(u,t) + sigma1())

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  c = 1 # 0.3

  def dxdt_return(x,t,u):

    #b2 = x/0.5
    #b1 = 1 - b2
    if x[1] >= 0.3:
      b1 = 0
      b2 = 1
    else:
      b1 = 1
      b2 = 0.05
    nn = 0
    #if x[0] < 0:
    #  nn = np.random.normal(0, 0.5) * np.array([1,0])
    return (b1*np.hstack([-1*x[0], 0.9*(x[1]+np.abs(x[0])*0.04)]) + #np.abs(
            b2*np.hstack([-2*x[0], -2*(x[1]-0.5)]) * c + inputs(u,t) + sigma2() + nn)

    if x[1] >= 0.3:
      return np.hstack([-2*x[0], -2*(x[1]-0.5)]) * c + inputs(u,t) + sigma2()
    else:
      return np.hstack([-1*x[0], 0.9*(x[1]+np.abs(x[0])*0.04)]) + inputs(u,t) + sigma2()

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])

  r1_state = 0

  # Simulate trials
  xs_trials1 = []
  msgs_trials = []
  switch_idxs = []
  for u in us:

    x1_ = x01
    xs1 = [x1_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r1_state == 0:

        dxdt_1 = dxdt_accum
        dx1_ = 0.1 * dt1 * dxdt_1(x1_,t,u)

        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][2].append(inputs(u,t))

      else:

        dxdt_ = dxdt_return
        dxdt_t = dxdt_(x1_,t,u)
        dx1_ = 0.1 * dt1 * dxdt_t # + r2_scale * x2_[1]

        msgs[0][0].append(dxdt_t)
        msgs[0][2].append(np.zeros(2))

      msgs[0][1].append(np.zeros(2))
      msgs[1][0].append(np.zeros(2))
      msgs[1][1].append(np.zeros(2))
      msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_
      xs1.append(x1_)

      if r1_state == 0 and i > 40: #(i > 50 or x1_ >) : #np.any(x2_ >= 2): #t > 50: #
        if np.random.rand() >= 0.5 or i > 50:
          if not accum_only:
            r1_state = 1
            switch_idxs.append(i+1)

    xs1 = np.stack(xs1) #.T
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    x01 = x1_ if history_effect else np.array([0.0, 0.5])

    # Reset r2 state
    r1_state = 0

  return xs_trials1, dt1, msgs_trials, switch_idxs


def simulate_xs_mr(us, T=75, feedback=False, r1s1_noise=0.0005,
                   r1s2_noise=0.035, r2_noise=0.1, r1_scale=0.6,
                   r2_scale=4.5*np.array([0.4, 0.7]),
                   history_effect=False):

  # Latent noise and dynamics
  sigma1 = lambda: np.random.normal(0, r1s1_noise)  # accum
  sigma2 = lambda: np.random.normal(0, r1s2_noise) * np.array([1, 0.1]) # only on 1st dim, return
  sigma = lambda: np.random.normal(0, r2_noise)  # region2
  A1 = np.array([[0.5, 0.0],[0.0, 0.5]])
  A2 = np.array([[0.1, 0.0],[0.0, 0.1]])

  # Inputs
  inputs = lambda u,t: u[np.where(ts==t)].flatten()

  # Accumulation dynamics gradient
  dxdt_accum = lambda x,t,u: (np.hstack([x[0]-np.power(x[0],3), -x[1]]) +
                              inputs(u,t) + sigma1())

  # Return dynamics gradient
  offset = np.array([0, 0.5])
  dxdt_return = lambda x,t,u: r1_scale*(-x + offset) + sigma2()

  dt1 = 1
  ts = np.arange(0, T*dt1, dt1)[1:]
  x01 = np.array([0.0, 0.5])
  x02 = np.array([0.0, 0.0])

  r2_state = 0

  # Simulate trials
  xs_trials1 = []
  xs_trials2 = []
  msgs_trials = []
  switch_idxs = []
  for u in us:

    x1_ = x01
    xs1 = [x1_]
    x2_ = x02
    xs2 = [x2_]
    msgs = [[[],[],[]], [[],[],[]]]

    for i, t in enumerate(ts):

      s = np.array([sigma(), sigma()])

      if r2_state == 0:

        dxdt_1 = dxdt_accum
        dx1_ = 0.1 * dt1 * dxdt_1(x1_,t,u)
        x2_ = A1 @ x2_ + (1-2*x1_[1]) + s

        msgs[0][0].append(dx1_ - inputs(u,t))
        msgs[0][1].append(np.zeros(2))
        msgs[0][2].append(inputs(u,t))

      else:

        dxdt_ = dxdt_return
        dxdt_t = dxdt_(x1_,t,u)
        dx1_ = dt1 * dxdt_t + r2_scale * x2_[1]
        s = np.array([sigma(), sigma()])
        x2_ = A2 @ x2_ + (1-2*x1_[1]) + s

        msgs[0][0].append(dxdt_t)
        msgs[0][1].append(r2_scale * x2_[1])
        msgs[0][2].append(np.zeros(2))

      msgs[1][0].append(np.array([1-2*x1_[1]] * 2))
      #print(A2@x2_ + s - x2_, (A2@x2_)[0], x2_[0])
      msgs[1][1].append(A2@x2_-x2_ + s)

      # Region 2 always gets no inputs
      msgs[1][2].append(np.zeros(2))

      x1_ = x1_ + dx1_
      xs1.append(x1_)
      xs2.append(x2_)

      if r2_state == 0 and i > 50: #np.any(x2_ >= 2): #t > 50: #
        r2_state = 1
        switch_idxs.append(i+1)

    xs1 = np.stack(xs1).T
    xs2 = np.stack(xs2).T
    msgs = np.stack(msgs)
    xs_trials1.append(xs1)
    xs_trials2.append(xs2)
    msgs_trials.append(msgs)

    # Next initial condition is last timepoint
    if history_effect:
      x01 = x1_
      x02 = x2_

    # Reset r2 state
    r2_state = 0

  return xs_trials1, xs_trials2, msgs_trials, dt1, switch_idxs


def generate_observations(xs_trials, obs_dims, seed, C_mat=None,
                          split_obs=False, rescale=True):
  """Project latents out to emissions."""
  np.random.seed(seed)

  latent_dims = xs_trials[0].shape[1]
  if C_mat is None:
    C_mat = np.random.rand(obs_dims, latent_dims)
  if split_obs:
    C_mat[int(obs_dims/2):, :int(latent_dims/2)] = 0
    C_mat[:int(obs_dims/2), int(latent_dims/2):] = 0
    # increase scale on second dim
    if rescale:
      C_mat[int(obs_dims/2):, int(latent_dims/2):] *= 3.5
  ys_trials = []
  for xs in xs_trials:
    ys = xs @ C_mat.T
    ys_trials.append(ys)
  return ys_trials, C_mat


def generate_observations_mr(xs_trials, obs_dims, num_regions, seed):
  """Project latents out to emissions."""
  np.random.seed(seed)

  print('gen obs', obs_dims)
  latent_dims = int(xs_trials[0].shape[1] / 2)
  C_mat1 = np.random.normal(0,1, (int(obs_dims/2), latent_dims))
  C_mat2 = np.random.normal(0,1, (int(obs_dims/2), latent_dims))

  ys_trials = []
  for xs in xs_trials:
    ys1 = xs[:,:latent_dims] @ C_mat1.T
    ys2 = xs[:,latent_dims:] @ C_mat2.T
    ys_trials.append(np.hstack([ys1, ys2]))
  return ys_trials, C_mat1, C_mat2


def generate_observations_mr_nle(xs_trials, obs_dims, num_regions, seed):
  """Project latents out to emissions."""
  np.random.seed(seed)

  print('gen obs nl', obs_dims)
  num_trials = len(xs_trials)
  T = xs_trials[0].shape[0]

  # Accumulator
  nn = int(obs_dims/2)
  x1_range = np.linspace(-0.85,0.85,nn-10)
  yjitters = [np.random.randn()/50 for _ in range(nn)]

  # Tile the x axis
  all_rvs2 = []
  for nidx in range(nn-10):
    rv = multivariate_normal([x1_range[nidx], yjitters[nidx]+0.05], [[0.01, 0.001], [0.001, 0.02]])
    all_rvs2.append(rv)

  # Last 10 neurons cover the y axis.
  x2_range = np.linspace(0.0,0.35,10)
  for nidx in range(10):
    rv = multivariate_normal([yjitters[nidx], x2_range[nidx]], [[0.02, 0.001], [0.001, 0.01]])
    all_rvs2.append(rv)

  ys_accum = np.zeros((num_trials,T,nn))
  for tr in range(num_trials):
    xx = xs_trials[tr][:,:2]
    for nidx in range(nn):
      yy = all_rvs2[nidx].pdf(xx)
      yy /= np.max(yy)
      yy *= 1.5
      ys_accum[tr,:,nidx] = yy

  # Controller
  nn = int(obs_dims/2)
  x1_range = np.linspace(-0.75,0.75,nn)
  x2_range = np.linspace(-0.15,0.15,nn)
  yjitters = [np.random.randn()/50 for _ in range(nn)]

  all_rvs = []
  for nidx in range(nn):
    rv = multivariate_normal([x1_range[nidx], yjitters[nidx]], [[0.01, 0.001], [0.001, 0.02]])
    all_rvs.append(rv)

  ys_cont = np.zeros((num_trials,T,nn))
  for tr in range(num_trials):
    xx = xs_trials[tr][:,3:]
    for nidx in range(nn):
      yy = all_rvs[nidx].pdf(xx)
      yy /= np.max(yy)
      yy *= 1.5
      ys_cont[tr,:,nidx] = yy

  ys_gb = np.concatenate([ys_accum, ys_cont], axis=-1)
  return ys_gb


def generate_grid(x1_range=[-1.5,1.6], x2_range=[-0.5,0.55], dx1=0.1, dx2=0.05):
  """For plotting contours and gradient."""
  # Continuous latent grid
  x1 = np.arange(*x1_range, dx1)
  x2 = np.arange(*x2_range, dx2)
  x1, x2 = np.meshgrid(x1, x2)
  return x1, x2


def get_potential_grad(x1, x2, bias):
  """Used for plotting."""
  # Energy potential state 1 (accumulator dynamics)
  def Vx1(x1, bias):
    v = np.zeros_like(x1)
    left = np.where(x1<0)
    right = np.where(x1>=0)
    v[left] = 0.5*np.power(x1[left],2) - 0.25*np.power(x1[left],4)
    v[right] = bias * (0.5*np.power(x1[right],2) - 0.25*np.power(x1[right],4))
    return v

  Vx2 = lambda x2: -0.5*np.power(x2,2)

  V_accum = Vx1(x1, bias) + Vx2(x2)

  # Gradient
  left = np.where(x1<0)
  right = np.where(x1>=0)
  dVdx1 = np.zeros_like(x1)
  dVdx1[left] = x1[left] - np.power(x1[left],3)
  dVdx1[right] = bias * (x1[right] - np.power(x1[right],3))
  dVdx2 = -x2

  # Energy potential state 2 (single return dynamics)
  a = 1
  offset = np.array([0, 0.5])
  Vx1_return = lambda x1: -0.5*a*np.power(x1,2) + 0.5*a*np.power(offset[0],2)
  Vx2_return = lambda x2: -0.5*a*np.power(x2,2) + 0.5*a*np.power(offset[1],2)
  V_return = Vx1_return(x1) + Vx2_return(x2)

  # Gradient
  dVdx1_return = -a*x1 + a*offset[0]
  dVdx2_return = -a*x2 + a*offset[1]

  return V_accum, dVdx1, dVdx2, V_return, dVdx1_return, dVdx2_return


def get_potential_grad_r2(x1, x2):
  """Used for plotting."""
  # Energy potential state 1 (controller accum dynamics)
  a = 0.1
  Vx1 = lambda x1: -(1-a)/2 * np.power(x1,2)
  Vx2 = lambda x2: -(1-a)/2 * np.power(x2,2)
  V_accum = Vx1(x1) + Vx2(x2)

  # Gradient
  dVdx1 = -(1-a)*x1
  dVdx2 = -(1-a)*x2

  # Energy potential state 2 (controller return dynamics)
  Vx1_return = Vx1
  Vx2_return = Vx2
  V_return = Vx1_return(x1) + Vx2_return(x2)

  # Gradient
  dVdx1_return = dVdx1
  dVdx2_return = dVdx2

  return V_accum, dVdx1, dVdx2, V_return, dVdx1_return, dVdx2_return


def get_potential_grad_sr(x1, x2, r1_scale):
  """Used for plotting."""
  # Energy potential state 1 (accumulator dynamics)
  Vx1 = lambda x1: 0.5*np.power(x1,2) - 0.25*np.power(x1,4)
  Vx2 = lambda x2: -0.5*np.power(x2,2)
  V_accum = 0.1 * (Vx1(x1) + Vx2(x2))
  # Gradient
  dVdx1 = 0.1 * (x1 - np.power(x1,3))
  dVdx2 = 0.1 * (-x2)

  # Energy potential state 2 (single return dynamics)
  a = r1_scale
  offset = np.array([0, 0.5])
  Vx1_return = lambda x1: -0.5*a*np.power(x1,2) + a*offset[0]*x1
  Vx2_return = lambda x2: -0.5*a*np.power(x2,2) + a*offset[1]*x2
  V_return = 1.4 * (Vx1_return(x1) + Vx2_return(x2))

  # Gradient
  dVdx1_return = 1.4 * (-a*x1 + a*offset[0])
  dVdx2_return = 1.4 * (-a*x2 + a*offset[1])

  return V_accum, dVdx1, dVdx2, V_return, dVdx1_return, dVdx2_return


def get_potential_grad_sr2(x1, x2, r1_scale):
  """Used for plotting."""
  # Energy potential state 1 (accumulator dynamics)
  Vx1 = lambda x1: 0.5*np.power(x1,2) - 0.25*np.power(x1,4)
  Vx2 = lambda x2: -0.5*np.power(x2,2)
  V_accum = Vx1(x1) + Vx2(x2)

  # Gradient
  dVdx1 = x1 - np.power(x1,3)
  dVdx2 = -x2

  # return1
  Vx1 = lambda x1: -(1/2)*np.power(x1,2)
  Vx2 = lambda x2, x1: 0.9 * ( 0.5*np.power(x2,2) + 0.04*np.abs(x1)*x2 ) #0.01*x2 # np.abs(x1)
  V_return = Vx1(x1) + Vx2(x2, x1)
  dVdx1_return = -1*x1
  dVdx2_return = 0.9 * (x2 + 0.04*np.abs(x1)) #0.01 #np.abs(x1)

  # return2
  c = 1
  V_ret2 = lambda x1,x2: -np.power(x1,2) - np.power(x2-0.5,2)
  V_return2 = V_ret2(x1,x2) * c
  dVdx1_return2 = -2*x1 * c
  dVdx2_return2 = -2*(x2-0.5) * c

  b1 = np.zeros_like(dVdx1_return)
  b1[np.where(x2<0.3)] = 1
  b1[np.where(b1==0)] = 0.05
  b2 = np.zeros_like(dVdx1_return2)
  b2[np.where(x2>=0.3)] = 1
  #b1 = 0 #.1 #0.99 #0.5
  #b2 = 1 #0.01
  dVdx1_return = b1*dVdx1_return + b2*dVdx1_return2
  dVdx2_return = b1*dVdx2_return + b2*dVdx2_return2

  b1 = np.zeros_like(V_return)
  b1[np.where(x2<0.3)] = 1
  b2 = np.zeros_like(V_return2)
  b2[np.where(x2>=0.3)] = 1
  V_return = b1*V_return + b2*V_return2

  return V_accum, dVdx1, dVdx2, V_return, dVdx1_return, dVdx2_return


def get_potential_grad_sr2_smooth(x1, x2, r1_scale):
  """Used for plotting."""
  # Energy potential state 1 (accumulator dynamics)
  Vx1 = lambda x1: 0.5*np.power(x1,2) - 0.25*np.power(x1,4)
  Vx2 = lambda x2: -0.5*np.power(x2,2)
  V_accum = Vx1(x1) + Vx2(x2)

  # Gradient
  dVdx1 = x1 - np.power(x1,3)
  dVdx2 = -x2

  # return1
  Vx1 = lambda x1: -(1/2)*np.power(x1,2)
  Vx2 = lambda x2, x1: 0.9 * ( 0.5*np.power(x2,2) + 0.04*np.abs(x1)*x2 ) #0.01*x2 # np.abs(x1)
  V_return = Vx1(x1) + Vx2(x2, x1)
  dVdx1_return = -1*x1
  dVdx2_return = 0.9 * (x2 + 0.04*np.abs(x1)) #0.01 #np.abs(x1)

  # return2
  c = 1
  V_ret2 = lambda x1,x2: -np.power(x1,2) - np.power(x2-0.5,2)
  V_return2 = V_ret2(x1,x2) * c
  dVdx1_return2 = -2*x1 * c
  dVdx2_return2 = -2*(x2-0.5) * c

  ps = powspace(0, 1, 1, 100)
  ls = np.linspace(0, 0.5, 100)
  def pick(ls, x):
    return ps[np.abs(ls-x).argmin()]
  pick = partial(pick, ls)
  b1 = np.vectorize(pick)(x2)
  b2 = 1 - b1
  dVdx1_return = b2*dVdx1_return + b1*dVdx1_return2
  dVdx2_return = b2*dVdx2_return + b1*dVdx2_return2
  V_return = b2*V_return + b1*V_return2

  return V_accum, dVdx1, dVdx2, V_return, dVdx1_return, dVdx2_return


def get_potential_grad_sr2a(x1, x2, r1_scale):
  """Used for plotting."""
  # Energy potential state 1 (accumulator dynamics)
  Vx1 = lambda x1: 0.5*np.power(x1,2) - 0.25*np.power(x1,4)
  Vx2 = lambda x2: -0.5*np.power(x2,2)
  V_accum = Vx1(x1) + Vx2(x2)

  # Gradient
  dVdx1 = x1 - np.power(x1,3)
  dVdx2 = -x2

  V_ret = lambda x1,x2: -np.power(x1,2) + -np.power(x2-0.5,2)
  V_return = V_ret(x1,x2)*0.2
  dVdx1_return = -2*x1 * 0.2
  dVdx2_return = -2*(x2-0.5) * 0.2

  a = 1 #-1
  return V_accum, dVdx1, dVdx2, a*V_return, a*dVdx1_return, a*dVdx2_return


def visualize_trajectories_old(x1, x2, V_accum, dVdx1, dVdx2, V_return,
                           dVdx1_return, dVdx2_return, xs_trials, T, dt1, dt2):
  """TBD"""
  fig, ax = plt.subplots(1,2, figsize=(12,5))
  ax[0].contour(x1, x2, -V_accum, 5, cmap="Greys")
  ax[0].quiver(x1, x2, dVdx1, dVdx2)
  ax[0].set_title('-V(x)', fontsize=14)
  ax[0].set_xlabel('x1', fontsize=14)
  ax[0].set_ylabel('x2', fontsize=14)
  for xs in xs_trials[:7]:
    if xs[0,50] > 0 :
      c = 'blue'
    else:
      c = 'green'
    ax[0].plot(xs[0,:50], xs[1,:50], alpha=0.5, lw=4, c=c)

  ax[0].set_ylim([0,0.5])
  ax[0].set_title('accumulation phase')
  ax[0].tick_params(axis='both', labelsize=12)

  ax[1].contour(x1, x2, -V_return, 5,cmap="Greys")
  ax[1].quiver(x1, x2, dVdx1_return, dVdx2_return)
  for xs in xs_trials[:7]:
    if xs[0,50] > 0 :
      c = 'blue'
    else:
      c = 'green'
    ax[1].plot(xs[0,50:], xs[1,50:], alpha=0.5, lw=4, c=c),
  ax[1].set_ylim([0,0.5])
  ax[1].set_yticks([])

  ax[1].set_xlabel('x1', fontsize=14)
  ax[1].set_title('return phase')
  ax[1].tick_params(axis='x', labelsize=12)

  for xs in xs_trials[:7]:
    ax[0].scatter(xs[0,0], xs[1,0], c='orange', s=550, marker="v", alpha=0.7)
    ax[1].scatter(xs[0,50], xs[1,50], c='orange', s=550, marker="^", alpha=0.7)

  plt.tight_layout()
  plt.show()

  fig, ax = plt.subplots(2,1, figsize=(6,3))
  for xs in xs_trials[:7]:
    if xs[0,50] > 0 :
      c = 'blue'
    else:
      c = 'green'
    ax[0].plot(xs[0,:], color=c, alpha=0.5)
    ax[1].plot(xs[1,:], color=c, alpha=0.5)
  ax[1].tick_params(axis='both', labelsize=12)
  ax[0].tick_params(axis='y', labelsize=12)
  ax[0].set_ylabel('x1', fontsize=14)
  ax[1].set_ylabel('x2', fontsize=14)
  ax[1].set_xlabel('t', fontsize=14)

  plt.show()


def visualize_trajectories(x1, x2, x1_2, x2_2, V_accum, dVdx1, dVdx2, V_return,
                           dVdx1_return, dVdx2_return, V_accum2, dVdx1_2, dVdx2_2,
                           V_return2, dVdx1_return2, dVdx2_return2,
                           xs_trials1, xs_trials2, zs, T, dt1, dt2,
                           title=False, switch_times=None, trial_types=None,
                           msgs=None, xs=None, ntrials=4):

  from mrsds.utils_plotting import reverse_colourmap
  import matplotlib.colors as colors
  from matplotlib.cm import get_cmap

  def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):

    new_cmap = colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
    cmap(np.linspace(minval, maxval, n)))
    return new_cmap

  clrs3 = ['grey', 'red']
  fig, ax = plt.subplots(1,2, figsize=(12,4))

  #ax[0].contour(x1, x2, -V_accum, 5,cmap="Greys", alpha=0.35)
  #ax[0].quiver(x1, x2, dVdx1, dVdx2, alpha=0.35, cmap='Greys')
  #ax[1].contour(x1, x2, -V_return, 5, cmap="Greys", alpha=0.35)
  #ax[1].quiver(x1, x2, dVdx1_return, dVdx2_return, cmap='Greys', alpha=0.35)
  nt = len(xs_trials1)
  for tr in range(nt):
    switch = switch_times[tr]
    for i in range(3):
      for t in range(T-1):
        msg_t = msgs[tr][0,i,t,:]
        xs_t = xs_trials1[tr][:,t]
        if np.linalg.norm(msg_t) > 0:
          # unit norm the msgs
          msg_t_unit = msg_t / (np.linalg.norm(msg_t) * 20)
          if i == 2:
            c = 'green' if msg_t_unit[0] < 0 else 'blue'
          else:
            c = clrs3[i]

          #ax_ = ax[1] if t >= switch else ax[0]
          ax_ = ax[0]
          ax_.arrow(*xs_t[:2], *msg_t_unit, head_width=0.02, head_length=0.025,
                    fc=c, ec=c, alpha=0.4, color=c)

    ax[0].plot(xs_trials1[tr][0,:], xs_trials1[tr][1,:], lw=4, alpha=0.2, color='grey')
    #ax[1].plot(xs_trials1[tr][0,switch:], xs_trials1[tr][1,switch:], lw=4, alpha=0.2, color='grey')

  for i, ax_ in enumerate(ax):
    ax_.set_xlim(-1.4, 1.4)
    ax_.set_ylim(-0.1, 0.6)
    c = 'darkgrey' # = 'green' if type_trials[tr] < 0 else 'blue'
    #for spine in ax_.spines.values():
    #  spine.set_visible(False)
    ax_.labelsize = 11
    ax_.tick_params(axis='both', labelsize=12)
    if i > 0:
      ax_.set_yticks([])

    for spine in ax_.spines.values():
      spine.set_edgecolor('grey')

  plt.tight_layout()
  plt.show()

  cmap = plt.get_cmap('Blues')
  blues_short_rv = reverse_colourmap(truncate_colormap(cmap, 0.05, 0.8))
  cmap = plt.get_cmap('Greens')
  greens_short_rv = reverse_colourmap(truncate_colormap(cmap, 0.05, 0.8))
  cmap = plt.get_cmap('Reds')
  #reds = truncate_colormap(cmap, 0.4, 1) #reverse_colourmap(
  #colors_red = [colors.to_hex(reds(i)) for i in range(0,150,10)]
  #trial_colors = colors_red
  trial_colors = ['orange', 'red', 'purple', 'sienna', 'fuchsia',
                  'peru', 'deeppink', 'gold', 'darkred', 'darkorange',
                  'indigo', 'chocolate', 'lightcoral', 'khaki', 'plum']

  if switch_times is None:
    switch_times = [50] * len(xs_trials)

  fig, ax = plt.subplots(2,2, figsize=(12,6), gridspec_kw={'height_ratios': [2.2, 1]} )
  #if title:
  #  fig.suptitle(title, fontsize=20)

  # Gradient flow fields and contours
  alpha = 0.5
  ax[0,0].contour(x1, x2, -V_accum, 5,cmap="Greys", alpha=alpha)
  ax[0,0].quiver(x1, x2, dVdx1, dVdx2,cmap="Greys", alpha=alpha)
  ax[0,1].contour(x1, x2, -V_return, 5,cmap="Greys", alpha=alpha)
  ax[0,1].quiver(x1, x2, dVdx1_return, dVdx2_return,cmap="Greys", alpha=alpha)

  for i, xs1 in enumerate(xs_trials1[:ntrials]):

    # Trajectory get colored by trial type
    switch = switch_times[i]
    if trial_types[i] == -1:
      c1 = 'green'
      c2 = 'lightgreen'
    else:
      c1 = 'blue'
      c2 = 'lightblue'
    ax[0,0].plot(xs1[0,:switch], xs1[1,:switch], alpha=0.5, lw=4, c=c1)
    ax[0,1].plot(xs1[0,switch:], xs1[1,switch:], alpha=0.5, lw=4, c=c1),

  for i, xs1 in enumerate(xs_trials1[:ntrials]):

    # Trajectory get colored by trial type
    switch = switch_times[i]
    if trial_types[i] == -1:
      c1 = 'green'
      c2 = 'lightgreen'
    else:
      c1 = 'blue'
      c2 = 'lightblue'
    # Start and end locations colored by trial index
    c = trial_colors[i] if i <= 15 else trial_colors[i%15]
    ax[0,0].scatter(xs1[0,0], xs1[1,0], c=c, s=200, marker="v", alpha=0.8)
    ax[0,0].scatter(xs1[0,switch], xs1[1,switch], c=c, s=200, marker="v", alpha=0.8)
    ax[0,1].scatter(xs1[0,switch], xs1[1,switch], c=c, s=200, marker="^", alpha=0.8, label=str(i+1))
    ax[0,1].scatter(xs1[0,-1], xs1[1,-1], c=c, s=200, marker="^", alpha=0.8)

  # Title limits etc.
  ax[0,0].set_xlabel('x1', fontsize=14)
  ax[0,0].set_ylabel('x2', fontsize=14)
  ax[0,0].set_ylim([0,0.5])
  #ax[0].set_title('accumulation phase')
  ax[0,0].tick_params(axis='both', labelsize=12)
  ax[0,1].set_ylim([0,0.5])
  ax[0,1].set_yticks([])
  ax[0,1].set_xlabel('x1', fontsize=14)
  #ax[1].set_title('return phase')
  ax[0,1].tick_params(axis='x', labelsize=12)
  ax[0,1].legend(fontsize=9.5, loc='upper right') #prop={'weight':'bold'},

  ax[0,0].axvline(0, ls='--', c='indigo')
  ax[0,1].axvline(0, ls='--', c='indigo')

  for _ in range(2):
    ax[0,_].set_xlim(-1.4, 1.4)
    ax[0,_].set_ylim(-0.1, 0.6)

  #  --- region 2 ---

  if xs_trials2 is not None:

    fig, ax = plt.subplots(1,2, figsize=(12,5))

    ax[1,0].contour(x1_2, x2_2, -V_accum2, 5,cmap="Greys")
    ax[1,0].quiver(x1_2, x2_2, dVdx1_2, dVdx2_2)
    ax[1,1].contour(x1_2, x2_2, -V_return2, 5,cmap="Greys")
    ax[1,1].quiver(x1_2, x2_2, dVdx1_return2, dVdx2_return2)

    for i, xs2 in enumerate(xs_trials2[:ntrials]):

      # Trajectory get colored by trial type
      switch = switch_times[i]
      if trial_types[i] == -1:
        c1 = 'green'
        c2 = 'lightgreen'
      else:
        c1 = 'blue'
        c2 = 'lightblue'
      ax[1,0].plot(xs2[0,:switch], xs2[1,:switch], alpha=0.45, lw=4, c=c1)
      ax[1,1].plot(xs2[0,switch:], xs2[1,switch:], alpha=0.45, lw=4, c=c1),

    for i, xs2 in enumerate(xs_trials2[:ntrials]):

      # Trajectory get colored by trial type
      switch = switch_times[i]
      if trial_types[i] == -1:
        c1 = 'green'
        c2 = 'lightgreen'
      else:
        c1 = 'blue'
        c2 = 'lightblue'
      # Start and end locations colored by trial index
      c = trial_colors[i] if i <= 15 else trial_colors[i%15]
      ax[1,0].scatter(xs2[0,0], xs2[1,0], c=c, s=200, marker=">", alpha=0.8)
      ax[1,0].scatter(xs2[0,switch], xs2[1,switch], c=c, s=200, marker=">", alpha=0.8)
      ax[1,1].scatter(xs2[0,switch], xs2[1,switch], c=c, s=200, marker="<", alpha=0.8, label=str(i+1))
      ax[1,1].scatter(xs2[0,-1], xs2[1,-1], c=c, s=200, marker="<", alpha=0.8)

    ax[1,0].set_xlabel('x1', fontsize=14)
    ax[1,0].set_ylabel('x2', fontsize=14)
    ax[1,0].tick_params(axis='both', labelsize=12)
    ax[1,1].set_yticks([])
    ax[1,1].set_xlabel('x1', fontsize=14)
    ax[1,1].tick_params(axis='x', labelsize=12)
    ax[1,0].set_ylim([-0.25,2.25])
    ax[1,1].set_ylim([-0.25,2.25])

    for ax_ in ax.flatten():
      for spine in ax_.spines.values():
        spine.set_edgecolor('grey')

    plt.tight_layout()
    plt.show()

    # 1Plot region1 and latent dims 1 and 2 vs time

    fig, ax = plt.subplots(4,1, figsize=(6,5),
                         gridspec_kw={'height_ratios':[1.2,1.2,1,1]})
    for i, xs in enumerate(xs_trials1[:ntrials]):
      if trial_types[i] == -1:
        c = 'green'
      else:
        c = 'blue'
      ax[0].plot(xs[0,:], color=c, alpha=0.5)
      ax[1].plot(xs[1,:], color=c, alpha=0.5)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].set_ylabel('x1', fontsize=14)
    ax[1].set_ylabel('x2', fontsize=14)
    ax[1].set_xlabel('t', fontsize=14)
    #plt.show()

    # Plot region1 and latent dims 1 and 2 vs time

    #fig, ax = plt.subplots(2,1, figsize=(6,3))
    for i, xs2 in enumerate(xs_trials2[:ntrials]):
      if trial_types[i] == -1:
        c = 'green'
      else:
        c = 'blue'
      ax[2].plot(xs2[0,:], color=c, alpha=0.5)
      ax[3].plot(xs2[1,:], color=c, alpha=0.5)
    ax[3].tick_params(axis='both', labelsize=12)
    ax[2].tick_params(axis='y', labelsize=12)
    ax[2].set_ylabel('x1', fontsize=14)
    ax[3].set_ylabel('x2', fontsize=14)
    ax[3].set_xlabel('t', fontsize=14)
    plt.show()


def make_result_plots(target_dir, args):

  xs_test, C_mat1, dat = args

  # mkdir for result plots if doesn't exit

  fix, ax = plt.subplots(1,2, figsize=(10,3))
  for tr in range(10):
    for i in range(2):
      ax[i].plot(np.arange(75), xs_test[tr][:,i])
      ax[0].set_title('x1')
      ax[1].set_title('x2')
      ax[0].set_xlabel('t')
      ax[1].set_xlabel('t')
  plt.tight_layout()
  plt.savefig(target_dir + 'sim-xs.png')

  proj = (C_mat1 @ xs_test[0].T)
  fix, ax = plt.subplots(1, 2, figsize=(10,3))
  for i in range(25):
    ax[0].plot(np.arange(75), proj[i,:])
  for i in range(25,50):
    ax[1].plot(np.arange(75), proj[i,:])
    ax[0].set_title('neurons 1:25')
    ax[1].set_title('neurons 25:50')
    ax[0].set_xlabel('t')
    ax[1].set_xlabel('t')
  plt.tight_layout()
  plt.savefig(target_dir + 'sim-proj-xs.png')

  plt.imshow(C_mat1, aspect='auto')
  plt.title('C mat')
  plt.savefig(target_dir + 'sim-cmat.png')

  nt = 5
  fig, ax = plt.subplots(nt,2, figsize=(10,10))
  for tr in range(nt):
    ax[tr,0].imshow(dat['ys_test'][tr,:].squeeze().T)
    ax[tr,1].imshow(dat['ys_recon_test'][tr,:].squeeze().T)
    if tr == nt-1:
      ax[tr,0].set_xlabel('t', fontsize=14)
      ax[tr,1].set_xlabel('t', fontsize=14)
    else:
      ax[tr,0].set_xticks([])
      ax[tr,1].set_xticks([])
    ax[tr,0].set_ylabel('neurons', fontsize=14)
    if tr == 0:
      ax[tr,0].set_title('true', fontsize=18)
      ax[tr,1].set_title('recon', fontsize=18)
  [ax_.tick_params(axis='both', labelsize=9) for ax_ in ax.flatten()]
  plt.tight_layout()
  plt.show()
