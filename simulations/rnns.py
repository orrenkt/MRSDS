
import math

import numpy as np
import numpy.random as npr
import numpy.linalg
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def generate_seq_driver_input(tData, N, leadTime, dtData, bumpStd):

  # set up sequence-driving network
  xBump = np.zeros((N, len(tData)))
  sig = bumpStd*N  # width of bump in N units
  norm_by = 2*sig ** 2
  cut_off = math.ceil(len(tData)/2) - int(len(tData)/10)
  for i in range(N):
    stuff = (i - sig - N*tData / (tData[-1]/2)) ** 2 / norm_by
    xBump[i, :] = np.exp(-stuff)
    xBump[i, cut_off:] = xBump[i, cut_off]

  hBump = np.log((xBump+0.01)/(1-xBump+0.01))
  hBump = hBump-np.min(hBump)
  hBump = hBump/np.max(hBump)

  newmat = np.tile(hBump[:, 1, np.newaxis], (1, math.ceil(leadTime/dtData)))
  hBump = np.concatenate((newmat, hBump), axis=1)

  return xBump, hBump


def generate_fixed_point_input(tData, N, leadTime, dtData, xBump):

  # set up fixed points driving network
  xFP = np.zeros((N, len(tData)))
  cut_off = math.ceil(len(tData)/2) + int(len(tData)/10)
  for i in range(N):
    front = xBump[i, int(len(tData)*0.01)] * np.ones((1, cut_off))
    back = xBump[i, int(len(tData)*0.3)] * np.ones((1, len(tData)-cut_off))
    xFP[i, :] = np.concatenate((front, back), axis=1)
  hFP = np.log((xFP+0.01)/(1-xFP+0.01))
  hFP = hFP - np.min(hFP)
  hFP = hFP/np.max(hFP)

  newmat = np.tile(hFP[:, 1, np.newaxis], (1, math.ceil(leadTime/dtData)))
  hFP = np.concatenate((newmat, hFP), axis=1)
  print('hfp', hFP.shape, len(tData), tData.shape)
  return hFP


def generate_nets(N, ga, gb, gc, fracInterReg, fracExternal, seed=0):

  np.random.seed(seed)

  Na = Nb = Nc = N

  # set up RNN A (chaotic responder)
  Ja = npr.randn(Na, Na) #+ np.eye(Na)
  Ja = (ga / math.sqrt(Na)) * Ja

  # set up RNN B (driven by sequence)
  Jb = npr.randn(Nb, Nb)
  Jb = (gb / math.sqrt(Na)) * Jb

  # set up RNN C (driven by fixed point)
  Jc = npr.randn(Nc, Nc)
  Jc = (gc / math.sqrt(Na)) * Jc

  def random_frac(N, Nfrac):
    rand_idx = npr.permutation(N)
    zeros = np.zeros((N,1))
    zeros[rand_idx[0:Nfrac]] = 1
    return zeros

  # build sparse connectivity between RNNs
  Nfrac = int(fracInterReg * N)
  w_A2B = random_frac(N, Nfrac)
  w_A2C = random_frac(N, Nfrac)
  w_B2A = random_frac(N, Nfrac)
  w_B2C = random_frac(N, Nfrac)
  w_C2A = random_frac(N, Nfrac)
  w_C2B = random_frac(N, Nfrac)

  # Sequence only projects to B, fixed points only to A. Also sparse
  Nfrac = int(fracExternal * N)
  w_Seq2B = random_frac(N, Nfrac)
  w_Fix2C = random_frac(N, Nfrac)

  return (Ja, Jb, Jc, w_A2B, w_A2C, w_B2A, w_B2C,
          w_C2A, w_C2B, w_Seq2B, w_Fix2C)


def simulate(mats, N, tData, dtData, ampInterReg, ampInB, ampInC,
             tau, hBump, hFP, seed=0, nls=[np.tanh, lambda x: x], noise=True): #False):

  np.random.seed(seed)

  Ja, Jb, Jc, w_A2B, w_A2C, w_B2A, w_B2C, w_C2A, w_C2B, w_Seq2B, w_Fix2C = mats

  # start from random state
  random_ic = lambda N: 2 * npr.rand(N, 1) - 1  # NOTE no tanh here.

  # Initial conditions for the 3 region nets.
  hCa, hCb, hCc = [random_ic(N) for _ in range(3)]

  # Generate time series simulated data
  T = len(tData)
  Ra, Rb, Rc = [np.zeros((N, T)) for _ in range(3)]
  Ra[:, 0, np.newaxis] = hCa
  Rb[:, 0, np.newaxis] = hCb
  Rc[:, 0, np.newaxis] = hCc

  (Raa, Rab, Rac, Rau1, Rau2,
   Rba, Rbb, Rbc, Rbu1, Rbu2,
   Rca, Rcb, Rcc, Rcu1, Rcu2) =  [np.zeros((T, N, 1)) for _ in range(15)]

  # We'll also collect messages
  messages = [[Raa, Rab, Rac, Rau1, Rau2],
              [Rba, Rbb, Rbc, Rbu1, Rbu2],
              [Rca, Rcb, Rcc, Rcu1, Rcu2]]

  # Dynamics and communications functions
  scale = dtData / tau
  nl = nls[0]
  Faa = lambda Ra_prev: scale * (nl(Ja.dot(Ra_prev)) - Ra_prev) + Ra_prev
  Fab = lambda Rb_prev: scale * nl(ampInterReg * w_B2A * Rb_prev)
  Fac = lambda Rc_prev: scale * nl(ampInterReg * w_C2A * Rc_prev)
  Fbb = lambda Rb_prev: scale * (nl(Jb.dot(Rb_prev)) - Rb_prev) + Rb_prev
  Fba = lambda Ra_prev: scale * nl(ampInterReg * w_A2B * Ra_prev)
  Fbc = lambda Rc_prev: scale * nl(ampInterReg * w_C2B * Rc_prev)
  Fbu1 = lambda hBump_prev: scale * nl(ampInB * w_Seq2B * hBump_prev)
  Fcc = lambda Rc_prev: scale * (nl(Jc.dot(Rc_prev)) - Rc_prev) + Rc_prev
  Fcb = lambda Rb_prev: scale * nl(ampInterReg * w_B2C * Rb_prev)
  Fca = lambda Ra_prev: scale * nl(ampInterReg * w_A2C * Ra_prev)
  Fcu2 = lambda hFP_prev: scale * nl(ampInC * w_Fix2C * hFP_prev)
  Fzero = lambda x: 0

  Fs = [[Faa, Fab, Fac, Fzero, Fzero],
        [Fba, Fbb, Fbc, Fbu1, Fzero],
        [Fca, Fcb, Fcc, Fzero, Fcu2]]

  # NOTE should make noise different in each neuron, ie 1000x1 noise vector
  if noise:
    levels = [0.001, 0.015, 0.02]
    Ns = [lambda : np.random.normal(0, levels[i])
          for i in range(3)]
  else:
    Ns = [lambda : np.zeros(1) for _ in range(3)]

  nl2 = nls[1]
  for t in range(1,T):

    # Chaotic responder region
    Ra_prev = Ra[:, t-1, np.newaxis]
    JRa = Faa(Ra_prev)
    Raa[t,:] = JRa - Ra_prev
    Rab[t,:] = Fab(Rb[:, t-1, np.newaxis])
    Rac[t,:] = Fac(Rc[:, t-1, np.newaxis])
    Ra[:, t, np.newaxis] = nl2(JRa + Rab[t,:] + Rac[t,:]) + Ns[0]()

    # Sequence driven region
    Rb_prev = Rb[:, t-1, np.newaxis]
    JRb = Fbb(Rb_prev)
    Rbb[t,:] = JRb - Rb_prev
    Rba[t,:] = Fba(Ra[:, t-1, np.newaxis])
    Rbc[t,:] = Fbc(Rc[:, t-1, np.newaxis])
    Rbu1[t,:] = Fbu1(hBump[:, t-1, np.newaxis])
    Rb[:, t, np.newaxis] = nl2(JRb + Rba[t,:] + Rbc[t,:] + Rbu1[t,:]) + Ns[1]()

    # Fixed point driven region
    Rc_prev = Rc[:, t-1, np.newaxis]
    JRc = Fcc(Rc_prev)
    Rcc[t,:] = JRc - Rc_prev
    Rcb[t,:] = Fcb(Rb[:, t-1, np.newaxis])
    Rca[t,:] = Fca(Ra[:, t-1, np.newaxis])
    Rcu2[t,:] = Fcu2(hFP[:, t-1, np.newaxis])
    Rc[:, t, np.newaxis] = nl2(JRc + Rcb[t,:]+ Rca[t,:] + Rcu2[t,:]) + Ns[2]()

  # package up outputs
  # normalize
  # NOTE: should take another look at this.
  messages[0] = [_/np.max(np.abs(Ra)) for _ in messages[0]]
  messages[1] = [_/np.max(np.abs(Rb)) for _ in messages[1]]
  messages[2] = [_/np.max(np.abs(Rc)) for _ in messages[2]]
  Ra = Ra/np.max(np.abs(Ra))
  Rb = Rb/np.max(np.abs(Rb))
  Rc = Rc/np.max(np.abs(Rc))

  return Ra, Rb, Rc, Fs, messages


def three_region_sim(number_units=100, ga=1.8, gb=1.5, gc=1.5, tau=0.1,
                     fracInterReg=0.01, #0.05,
                     ampInterReg=0.04, #0.02,
                     fracExternal=0.5, ampInB=1, ampInC=-1, dtData=0.01,
                     T=10, leadTime=2, bumpStd=0.2, plotsim=False, seed=0,
                     ntrials=1, plot_tr=0):
    """
    Generates a simulated dataset with three interacting regions. Modified from
    Perich MG et al. Inferring brain-wide interactions using data-constrained
    recurrent neural network models.     %

    Parameters
    ----------

    number_units: int
        number of units in each region
    ga: float
        chaos parameter for Region A
    gb: float
        chaos parameter for Region B
    gc: float
        chaos parameter for Region C
    tau: float
        decay time constant of RNNs
    fracInterReg: float
        fraction of inter-region connections
    ampInterReg: float
        amplitude of inter-region connections
    fracExternal: float
        fraction of external inputs to B/C
    ampInB: float
        amplitude of external inputs to Region B
    ampInC: float
        amplitude of external inputs to Region C
    dtData: float
        time step (s) of the simulation
    T: float
        total simulation time
    leadTime: float
        time before sequence starts and after FP moves
    bumpStd: float
        width (in frac of population) of sequence/FP
    plotSim: bool
        whether to plot the results
    """
    np.random.seed(seed)

    tData = np.arange(0, (T + dtData), dtData)

    # for now it only works if the networks are the same size
    N = Na = Nb = Nc = number_units

    mats = generate_nets(N, ga, gb, gc, fracInterReg, fracExternal, seed)
    Ja, Jb, Jc, w_A2B, w_A2C, w_B2A, w_B2C, w_C2A, w_C2B, w_Seq2B, w_Fix2C = mats

    # generate external inputs
    xBump, hBump = generate_seq_driver_input(tData, N, leadTime, dtData, bumpStd)
    hFP = generate_fixed_point_input(tData, N, leadTime, dtData, xBump)

    # package up outputs -- NOTE should also take another look at this normalization.
    Rseq = hBump.copy()
    Rfp = hFP.copy()
    Rseq = Rseq/np.max(Rseq)
    Rfp = Rfp/np.max(Rfp)
    Rnull = np.zeros_like(Rfp)

    # add the lead time NOTE - this and hFP can be off by one.
    extratData = np.arange(tData[-1] + dtData, T + leadTime, dtData)
    tData = np.concatenate((tData, extratData))

    # NOTE this assumes we always simulate with the same input, might want to change
    Ra = []
    Rb = []
    Rc = []
    msgs = []
    Ru = []
    for i in range(ntrials):
      Ra_, Rb_, Rc_, Fs, msgs_ = simulate(mats, N, tData, dtData, ampInterReg, ampInB,
                                          ampInC, tau, hBump, hFP, seed=i)
      Ra.append(Ra_)
      Rb.append(Rb_)
      Rc.append(Rc_)
      msgs.append(msgs_)
      Ru.append(np.concatenate([Rnull, Rseq, Rfp], axis=0))

    Ra = np.array(Ra)
    Rb = np.array(Rb)
    Rc = np.array(Rc)
    Ru = np.array(Ru)
    msgs = np.array(msgs)

    out_params = {}
    out_params['Na'] = Na
    out_params['Nb'] = Nb
    out_params['Nc'] = Nc
    out_params['ga'] = ga
    out_params['gb'] = gb
    out_params['gc'] = gc
    out_params['tau'] = tau
    out_params['fracInterReg'] = fracInterReg
    out_params['ampInterReg'] = ampInterReg
    out_params['fracExternal'] = fracExternal
    out_params['ampInB'] = ampInB
    out_params['ampInC'] = ampInC
    out_params['dtData'] = dtData
    out_params['T'] = T
    out_params['leadTime'] = leadTime
    out_params['bumpStd'] = bumpStd

    out = {}
    out['Ra'] = Ra
    out['Rb'] = Rb
    out['Rc'] = Rc
    out['Rseq'] = Rseq
    out['Rfp'] = Rfp
    out['Ru'] = Ru
    out['tData'] = tData
    out['Ja'] = Ja
    out['Jb'] = Jb
    out['Jc'] = Jc
    out['w_A2B'] = w_A2B
    out['w_A2C'] = w_A2C
    out['w_B2A'] = w_B2A
    out['w_B2C'] = w_B2C
    out['w_C2A'] = w_C2A
    out['w_C2B'] = w_C2B
    out['w_Fix2C'] = w_Fix2C
    out['w_Seq2B'] = w_Seq2B
    out['params'] = out_params
    out['msgs'] = msgs
    out['Fs'] = Fs

    ys = np.swapaxes(np.hstack([out['Ra'], out['Rb'], out['Rc']]), 1,2)

    # Since the input is 1 per neuron dim we project onto PCs.
    pca_fp = PCA(n_components=1)
    rfp_pc = pca_fp.fit_transform(out['Rfp'].T)
    pca_seq = PCA(n_components=3)
    rseq_pc = pca_seq.fit_transform(out['Rseq'].T)
    us = np.tile(np.hstack([rfp_pc, rseq_pc]), (ntrials,1,1))
    print('rnn sim', ys.shape, us.shape, out['Rseq'].shape[0], ntrials)

    # NOTE hack to catch off by one error, NEED TO FIX
    us = us[:ntrials,:ys.shape[1],:]

    #us = np.tile(np.vstack([out['Rseq'], out['Rfp']]), (out['Rseq'].shape[0],1,1))
    #us = np.swapaxes(us, 1,2)

    #if plotsim is True:
    #  plot_sim(tData, out, out_params, tr=plot_tr)
    #return out
    return ys, us, msgs, Fs


def get_potential_grad(sim):

  # NOTE WIP not working
  raise ValueError('not implemented')

  for l, data_ in enumerate([sim['Ra'], sim['Rb'], sim['Rc']]):
    num_trials, num_neurons, nt = data_.shape
    data_flat = np.moveaxis(data_, 0,-1).reshape(num_neurons, nt*num_trials).T

    pca = PCA(n_components=2)
    pca.fit(data_flat)
    PCs = pca.components_
    data_pc = pca.transform(data_flat)

    mult = 1.2
    minx, maxx, miny, maxy = (np.min(data_pc[:,0]) * mult,
                              np.max(data_pc[:,0]) * mult,
                              np.min(data_pc[:,1]) * mult,
                              np.max(data_pc[:,1]) * mult)
    gridlen = 15
    gridx = np.linspace(minx,maxx,gridlen)
    gridy = np.linspace(miny,maxy,gridlen)
    grid = np.vstack([gridx, gridy])
    grid = np.vstack([_.flatten() for _ in np.meshgrid(*grid)])
    grid_up = pca.components_[:2,:].T @ grid
    grads = sim['Fs'][l][l](grid_up) - grid_up
    grads_pc = pca.components_[:2,:] @ grads

    for i in range(gridlen**2):

        grad = grads_pc[:,i].squeeze()
        grad_norm = grad / np.linalg.norm(grad) * 0.5
        xy = grid[:,i].squeeze()
        #ax.arrow(xy[0], xy[1], )
        hw = 0.4
        hl = 0.5
        if l == 0:
            hw *= 1.7
            hl *= 1.7
        elif l == 1:
            hw *= 0.7
            hl *= 0.7
        #ax[l].arrow(*xy, *grad_norm, head_width=hw, head_length=hl,
        #            alpha=0.5, color='black')

  V_1 = V_2 = V_3 = 0
  return V_1, *dVdxs[0], V_2, *dVdxs[1], V_3, *dVdxs[2], grid_up


def plot_sim(tData, out, out_params, tr=0):

  N = out_params['Na']

  fig = plt.figure(figsize=[10, 10])
  fig.tight_layout()
  fig.subplots_adjust(hspace=0.4, wspace=0.3)
  plt.rcParams.update({'font.size': 6})

  ax = fig.add_subplot(4, 3, 1)
  ax.pcolormesh(tData, range(N), out['Ra'][tr,:])
  ax.set_title('RNN A - g={}'.format(out_params['ga']))

  ax = fig.add_subplot(4, 3, 2)
  ax.pcolormesh(range(N), range(N), out['Ja'])
  ax.set_title('DI matrix A')

  ax = fig.add_subplot(4, 3, 3)
  for _ in range(3):
    idx = np.random.randint(0, N-1)
    ax.plot(tData, out['Ra'][tr, idx, :])
  ax.set_ylim(-1, 1)
  ax.set_title('units from RNN A')

  ax = fig.add_subplot(4, 3, 4)
  ax.pcolormesh(tData, range(N), out['Rb'][tr,:])
  ax.set_title('RNN B - g={}'.format(out_params['gb']))

  ax = fig.add_subplot(4, 3, 5)
  ax.pcolormesh(range(N), range(N), out['Jb'])
  ax.set_title('DI matrix B')

  ax = fig.add_subplot(4, 3, 6)
  for _ in range(3):
    idx = np.random.randint(0, N-1)
    ax.plot(tData, out['Rb'][tr, idx, :])
  ax.set_ylim(-1, 1)
  ax.set_title('units from RNN B')

  ax = fig.add_subplot(4, 3, 7)
  ax.pcolormesh(tData, range(N), out['Rc'][tr,:])
  ax.set_title('RNN C - g={}'.format(out_params['gc']))

  ax = fig.add_subplot(4, 3, 8)
  ax.pcolormesh(range(N), range(N), out['Jc'])
  ax.set_title('DI matrix C')

  ax = fig.add_subplot(4, 3, 9)
  for _ in range(3):
    idx = np.random.randint(0, N-1)
    ax.plot(tData, out['Rc'][tr, idx, :])
  ax.set_ylim(-1, 1)
  ax.set_title('units from RNN C')

  # NOTE hack to handle off by one issue on Rfp and Rseq, not critical
  ax = fig.add_subplot(4, 3, 10)
  ax.pcolormesh(tData, range(N), out['Rfp'][:,:len(tData)])
  ax.set_title('Fixed Point Driver')
  ax = fig.add_subplot(4, 3, 11)
  ax.pcolormesh(tData, range(N), out['Rseq'][:,:len(tData)])
  ax.set_title('Sequence Driver')
  plt.show()
