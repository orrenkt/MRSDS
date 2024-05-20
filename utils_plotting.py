
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib as mpl
import numpy as np
import matplotlib.colors as colors
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")


class MidpointNormalize(colors.Normalize):
  """
  Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
  e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
  """
  def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
    self.midpoint = midpoint
    colors.Normalize.__init__(self, vmin, vmax, clip)

  def __call__(self, value, clip=None):
    # I'm ignoring masked values and all kinds of edge cases to make a
    # simple example...
    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def get_colors():
  color_names = ["windows blue",
                 "red",
                 "amber",
                 "faded green",
                 "dusty purple",
                 "orange",
                 "clay",
                 "pink",
                 "greyish",
                 "mint",
                 "light cyan",
                 "steel blue",
                 "forest green",
                 "pastel purple",
                 "salmon",
                 "dark brown"]
  colors_ = sns.xkcd_palette(color_names)
  return colors_


def plot_norms(msg_input_norms, tr):

  num_regions = msg_input_norms.shape[2]
  num_timepoints = msg_input_norms.shape[1]

  fig, ax = plt.subplots(num_regions,num_regions+1,figsize=(10,5))
  clrs = np.flip(sns.color_palette("Greys", 16)[::2])[:6]

  for i in range(num_regions):
    for j in range(num_regions+1):

      ax[i,j].plot(np.arange(num_timepoints), msg_input_norms[tr,:,i,j])
      ax[i,j].set_yticks([])
      ax[i,j].set_xticks([])
      ax[i,j].set_facecolor("white")

      [ax[i,j].spines[_].set_color('lightgrey')
       for _ in ax[i,j].spines]

      ax[i,j].spines['right'].set_visible(False)
      ax[i,j].spines['top'].set_visible(False)

      if j != 0:
        ax[i,j].spines['left'].set_visible(False)

      if i != 2:
        ax[i,j].spines['bottom'].set_visible(False)

      if i == 0:
        if j == num_regions+1:
          lbl = 'towers'
          fw = 'bold'
      ax[i,j].axhline(0, lw=1, ls='--', color='grey')
  plt.show()


def plot_overlaid(x1_, x2_, dVdx1, dVdx2, V_accum, dVdx1_return, dVdx2_return, V_return,
                  inf_dvdxs, zs_train, qzs, xs_, xs_pred_trans, nts=np.arange(1),
                  contours=True, c1='brown', c2='darkred', num_states=1,
                  lims=[(0,2.5),(0,2.5)]):

  fig, ax = plt.subplots(1,2,figsize=(12,5))
  for tr in nts:
    ax[0].plot(xs_[tr,:,0], color=c1, alpha=0.6)
    ax[0].plot(xs_pred_trans[tr,:,0], color=c2, alpha=0.6) #, ls='--')
    ax[1].plot(xs_[tr,:,1], color=c1, alpha=0.6)
    ax[1].plot(xs_pred_trans[tr,:,1], color=c2, alpha=0.6) #, ls='--')
  ax[0].set_ylabel('x1')
  ax[1].set_ylabel('x2')
  ax[0].set_ylim(*lims[0])
  ax[1].set_ylim(*lims[1])
  plt.show()

  fig, ax = plt.subplots(1,2,figsize=(10,5))

  grads = [(V_accum, dVdx1, dVdx2, *inf_dvdxs[0])]
  if len(inf_dvdxs) == 2:
    grads.append((V_return, dVdx1_return, dVdx2_return, *inf_dvdxs[1]))
  print('grads len', len(grads))
  for i, (V, dvdx1, dvdx2, dvdx1_inf, dvdx2_inf) in enumerate(grads):
    print(i)
    if contours:
      ax[i].contour(x1_, x2_, -V, 5,cmap="Greys", levels=6, alpha=0.5)
    ax[i].quiver(x1_, x2_, dvdx1, dvdx2, angles='xy', color='grey',
                 width=0.005, alpha=0.9)
    ax[i].quiver(x1_, x2_, dvdx1_inf, dvdx2_inf, angles='xy',
                 color='steelblue', alpha=0.85, width=0.005)

    for tr in nts:
      c = c1
      switchpts = np.hstack([0, np.where(zs_train[tr][:-1]!=zs_train[tr][1:])[0],
                             len(zs_train[tr])-1])
      states_ = zs_train[tr]
      # NOTE hack for first timepoint
      #print('switchpts inf', switchpts)
      switch_states = states_[switchpts[1:]]
      starts_ends = list(zip(switchpts[:-1], switchpts[1:]))
      print(i, tr, switch_states, starts_ends)
      if len(starts_ends) == 1:
        ax[i].plot(xs_[tr,:,0], xs_[tr,:,1], alpha=0.5, color=c, lw=4)
      else:
        for j, (s, e) in enumerate(starts_ends):
          if j == 0 and i == 0:
            ax[i].scatter(xs_[tr,e,0], xs_[tr,e,1], color='yellow', s=100, alpha=0.8)
          elif j == 1 and i == 1:
            ax[i].scatter(xs_[tr,s,0], xs_[tr,s,1], color='red', s=100, alpha=0.8)
          state_ = switch_states[j]
          if state_ == i:
            ax[i].plot(xs_[tr,s:e,0], xs_[tr,s:e,1], alpha=0.5, color=c, lw=4)

      c = c2
      states_ = np.argmax(qzs[tr,:,:], axis=-1)
      # NOTE hack for first timepoint
      switchpts = np.hstack([0, np.where(states_[1:-1]!=states_[2:])[0], len(states_)-1])
      #print('switchpts inf', switchpts)
      switch_states = states_[switchpts[1:]]
      starts_ends = list(zip(switchpts[:-1], switchpts[1:]))
      print(i, tr, switch_states, starts_ends)
      if len(starts_ends) == 1:
        ax[i].plot(xs_pred_trans[tr,:,0], xs_pred_trans[tr,:,1],
                alpha=0.5, color=c, lw=4)
      else:
        for j, (s, e) in enumerate(starts_ends):
          state_ = switch_states[j]
          if state_ == i:
            ax[i].plot(xs_pred_trans[tr,s:e,0], xs_pred_trans[tr,s:e,1],
                    alpha=0.5, color=c, lw=4)
    for child in ax[i].get_children():
      if isinstance(child, mpl.spines.Spine):
        child.set_color('grey')
        child.set_linewidth(1)

  ax[0].set_xlim(*lims[0])
  ax[0].set_ylim(*lims[1])
  ax[1].set_xlim(*lims[0])
  ax[1].set_ylim(*lims[1])

  plt.tight_layout()
  plt.show()
  return
  #plt.close()

    #if i == num_states-1:
    #  print('ret', i, num_states)
    #  return


def plot_xs(xs_test_, xs_test_pred_trans):

  # Need to also apply transform to messages sent from left, right towers.

  # --- Overview ---
  fig, ax = plt.subplots()
  for i in range(10):
    if xs_test_[i,0,0] <= 0:
      c = 'green'
      c2 = 'darkgreen'
    else:
      c = 'blue'
      c2 = 'darkblue'
    plt.scatter(xs_test_[i,0,0],xs_test_[i,0,1], color=c, alpha=0.7, s=100)
    plt.scatter(xs_test_pred_trans[i,0,0], xs_test_pred_trans[i,0,1],
                alpha=0.7, s=100, facecolors='none', edgecolors=c2) #color=c2,
    plt.plot([xs_test_[i,0,0],xs_test_pred_trans[i,0,0]],
             [xs_test_[i,0,1],xs_test_pred_trans[i,0,1]], color='grey',
              linestyle="--", alpha=1)

    plt.plot(xs_test_[i,:,0], xs_test_[i,:,1], lw=2.5, color=c, alpha=0.5)
    plt.plot(xs_test_pred_trans[i,:,0], xs_test_pred_trans[i,:,1],
             color=c2, lw=2, alpha=0.9, ls='--')

  ax.tick_params(color='dimgrey', labelcolor='dimgrey')
  for spine in ax.spines.values():
    spine.set_edgecolor('dimgrey')

  plt.ylim(-0.05, 0.55)
  plt.axvline(0, color='black', alpha=0.3)
  plt.title('region 1')
  plt.show()
  #plt.close()

  # --- Zoomed in ---

  for i in range(10):
    if xs_test_[i,0,0] <= 0:
      c = 'green'
      c2 = 'darkgreen'
    else:
      c = 'blue'
      c2 = 'darkblue'
    plt.scatter(xs_test_[i,0,0],xs_test_[i,0,1], color=c, alpha=0.7, s=100)
    plt.scatter(xs_test_pred_trans[i,0,0], xs_test_pred_trans[i,0,1],
                alpha=0.7, s=100, facecolors='none', edgecolors=c2) #color=c2,
    plt.plot([xs_test_[i,0,0],xs_test_pred_trans[i,0,0]],
             [xs_test_[i,0,1],xs_test_pred_trans[i,0,1]], color='grey',
              linestyle="--", alpha=1)

    plt.plot(xs_test_[i,:,0], xs_test_[i,:,1], color=c, alpha=0.3)
    plt.plot(xs_test_pred_trans[i,:,0], xs_test_pred_trans[i,:,1],
             color=c2, alpha=0.3, ls='--')

  plt.axvline(0, color='black', alpha=0.3)
  plt.ylim(0.43, 0.51)
  plt.xlim(-0.5, 0.5)
  plt.title('region 1 zoomed in')
  plt.show()
  #plt.close()

  # --- Projected onto x1 ---

  for i in range(10):
    if xs_test_[i,0,0] <= 0:
      c = 'green'
      c2 = 'darkgreen'
    else:
      c = 'blue'
      c2 = 'darkblue'
    plt.scatter(xs_test_[i,0,0],0, color=c, alpha=0.7, s=100)
    plt.scatter(xs_test_pred_trans[i,0,0], 0,
                alpha=0.7, s=100, facecolors='none', edgecolors=c2) #color=c2,
    plt.plot([xs_test_[i,0,0],xs_test_pred_trans[i,0,0]], [0,0], color='grey',
              linestyle="--", alpha=.5)
  plt.axvline(0, color='black', alpha=0.3)
  plt.title('x1')
  plt.show()
  #plt.close()


def plot_qzs(qzs):
  for i in range(3):
    plt.figure(figsize=(10,2))
    for j in range(1):
      plt.plot(np.arange(qzs.shape[1]), qzs[i,:,j], alpha=0.9)
    for j in range(1):
      plt.plot(zs_test[i], alpha=0.9)
    plt.show()
    #plt.close()

    fig, ax = plt.subplots(2,1, figsize=(5,3))
    ax[0].imshow(zs_test[i][1:-1][np.newaxis,:], aspect='auto')
    ax[0].set_xticks([])
    ax[1].imshow(qzs[i,:-1,j][np.newaxis,1:], aspect='auto')
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    plt.show()
    #plt.close()


def plot_ys(ys_true, ys_inf, ys_gen, tr=0):
  fig, ax = plt.subplots(3, 1, figsize=(6,6))
  ax[0].imshow(ys_true[tr,:].squeeze().T)
  ax[1].imshow(ys_inf[tr,:].squeeze().T)
  ax[2].imshow(ys_gen[tr,:].squeeze().T)
  ax[2].set_xlabel('t', fontsize=14)
  ax[0].set_xticks([])
  ax[1].set_xticks([])
  ax[0].set_ylabel('neurons', fontsize=14)
  ax[1].set_ylabel('inf', fontsize=14)
  ax[2].set_ylabel('gen', fontsize=14)
  #ax[0].set_title('true', fontsize=18)
  #ax[1].set_title('inf', fontsize=18)
  #ax[2].set_title('gen', fontsize=18)
  [ax_.tick_params(axis='both', labelsize=9) for ax_ in ax.flatten()]
  plt.tight_layout()
  plt.show()
  #plt.close()


def plot_neurons_hist(ys):
  """Visualize values per neurons for 15 random trials."""
  fig, ax = plt.subplots(1,1, figsize=(10,3))
  ntrials = 25
  ys_stacked = np.vstack(np.random.sample(ys, ntrials))
  [ax.hist(ys_stacked[:,i], alpha=0.5) for i in range(ys_stacked.shape[1])]
  plt.tight_layout()
  plt.show()


def plot_data_pc_var(ys, region_sizes, region_names, clrs=None):

  npcs = 40
  pcas = []
  starts = np.cumsum([0] + list(region_sizes))
  for i in range(len(region_sizes)):
    ys_i = np.vstack([ys[_][:,starts[i]:starts[i+1]] for _ in range(63)])
    pca = PCA(n_components=npcs)
    pca.fit(ys_i)
    pcas.append(pca)

  #clrs_ = ['darkred','darkblue','darkgreen']
  plt.figure(figsize=(10,6))
  for i, pca in enumerate(pcas):
    plt.plot(np.cumsum(pca.explained_variance_ratio_),
             label=region_names[i], alpha=0.7, lw=5, color=clrs[i]) #olors_[i])
  #plt.axvline(20, color='grey', linestyle='--', lw=3)
  #plt.axvline(3, color='grey', linestyle='--', lw=3)
  plt.legend(fontsize=11, ncol=2, frameon=False)
  plt.ylabel('var')
  plt.xlabel('PC')
  plt.tight_layout()
  #plt.savefig('figs/pca.svg')
  plt.show()
