# -*- coding: utf-8 -*-

"""
.. module:: daco_misc
   :platform: Unix, macOS
   :synopsis: Miscellaneous functions for comparing datasets.

.. moduleauthor:: Jon Vegard Sparre
                  Robindra Prabhu
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def compare_corr_diff(corr_benchmark, corr_others=[], identificators=[]):
  """Function for comparing diff of correlations between synth. and real data
  sets. Two or more correlation matrices are loaded, one of them is treated
  as a benchmark and the others are compared to this one.

  Parameters
  ----------
    corr_benchmark : str
      path to benchmark correlation matrix.
    corr_others : list
      paths to other correlation matrices as a list of strings
    identificators : list
      list of strings which indicates what is different from one
      diff to the next.

  Returns
  -------
    dict with results?
  """

  # Checking whether the files exists.
  if not os.path.exists(corr_benchmark):
    print(f"Could not find {corr_benchmark}")
    return FileNotFoundError

  for path in corr_others:
    if not os.path.exists(corr_benchmark):
      print(f"Could not find {path}")
      return FileNotFoundError

  correlations = {}
  with open(corr_benchmark, 'rb') as f:
    correlations['benchmark'] = pickle.load(f)

  diff_compare = {}
  for path, id_ in zip(corr_others, identificators):
    with open(path, 'rb') as f:
      correlations[id_] = pickle.load(f)
    diff_compare[id_] = np.abs(correlations['benchmark'] - correlations[id_])

  return diff_compare

def plot_compare_corr_diff(diff_compare, identificators, filename_suffix=''):
  """Plotting the results from `compare_corr_diff`.
  
  Parameters
  ----------
    diff_compare : dict
      dict with the absolute difference between the correlation matrices.
    identificators : list
      list of strings which indicates what is different from one
      diff to the next.
  """

  # Defining layout of canvas
  num_cols = 3
  num_rows = 3
  num_plots_in_canvas = num_cols*num_rows
  num_diffs = len(diff_compare) - 1
  num_figures = round((num_diffs)/num_plots_in_canvas)

  for canvas_fig in range(0,num_figures):
    gs = matplotlib.gridspec.GridSpec(num_rows, num_cols)

    # Creating figure instance and looping through batch of variables in
    # dataframe.
    fig = plt.figure(0, figsize=(15, 4*num_rows))
    i   = 0
    # Looping through all variables in dataframes
    for diff in identificators[num_plots_in_canvas*canvas_fig:num_plots_in_canvas*(canvas_fig+1)]:
      ax = fig.add_subplot(gs[i])
      i += 1
      mask = np.zeros_like(diff_compare[diff], dtype=np.bool)
      mask[np.triu_indices_from(mask)] = True
      cmap = sns.diverging_palette(240, 10, as_cmap=True)
      ax2 = sns.heatmap(diff_compare[diff], mask=mask, cmap=cmap, vmax=1, vmin=-1,
                square=True,
                linewidths=.5, cbar=1, cbar_kws={"fraction": 0.05,
                                                  "shrink": 0.5,
                                                  "orientation": "vertical"})
    
    plt.tight_layout()
    # TODO a path to folder for saving files must be given.
    # if canvas_fig == 0:
    #   plt.savefig(self.file_dir + 'canvas_{}.png'.format(filename_suffix))
    # elif canvas_fig > 0:
    #   plt.savefig(self.file_dir + 'canvas_{}_{}.png'.format(filename_suffix, canvas_fig))
    plt.show()
    plt.close()