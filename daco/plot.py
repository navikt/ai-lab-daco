#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module:: plot
   :platform: Unix, macOS
   :synopsis: Plotting module for DACO.

.. moduleauthor:: Jon Vegard Sparre
                  Robindra Prabhu

.. todo::
    - plotting/smart plotting, i.e. show only anomalies
    - pull plots based on the output from pd.dataframe.describe
    - allow setting range, density, binning, etc. for each variable manually?
    - add legends on boxplot for distance metrics
    - log-axes in plots
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scipy, os
from scipy import stats

class plot:
  """ Class for plotting the metrics found in :class:`daco`.
  """

  def __init__(self):
    pass

  def plotDistanceMetrics(self):
    """Plot boxplot of the distance metrics Kullback-Leibler, Bhattacharyya, and
    Hellinger for all numerical variables.
    """
    kl_array, bha_array, hel_array = self._findAndNormalizeDistances()

    fig = plt.figure(figsize=(9,6))
    # TODO bytte til seaborn --> må antakeligvis legge arrays over inn i en dataframe
    bp = plt.boxplot([kl_array, bha_array, hel_array], showmeans=True)
    plt.title('Boxplot av distansemetrikker på tvers av variable')
    plt.xticks([1, 2, 3], ['Kullback-\nLeibler', 'Bhattacharyya', 'Hellinger'])
    plt.show()

  def plotCorrelation(self, xlabel="", ylabel="", title="", filename="correlations", annotation=True):
    """Plotting correlations between numerical columns in a dataframe and saving
    as PDF-file.
    
    :param xlabel: label på x-aksen
    :type xlabel: str
    :param ylabel: label på y-aksen
    :type ylabel: str
    :param title: plottittel
    :type title: str
    :param filename: name of plot file
    :type filename: str
    """

    df1       = self.df1
    file_dir  = self.file_dir
    
    corr = df1.corr()                            # Pandasfunksjonalitet finner korrelasjonene.
    mask = np.zeros_like( corr, dtype=np.bool )  # Lager matrise som setter øvre triangel i X_df
    mask[ np.triu_indices_from( mask ) ] = True  # til null for å gjøre korrelasjonsplottet enklere.
    cmap = sns.diverging_palette( 240, 10, as_cmap=True )
    
    plt.figure( figsize=( 10, 10 ) )
    ax0 = sns.heatmap( corr, mask=mask, cmap=cmap, vmin=-1
                    , vmax=1, square=True, linewidths=.5, cbar=1
                    , cbar_kws={ "fraction": .05, "shrink": .5, "orientation": "vertical" }
                    , annot=annotation )
    ax0.set_title(title)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(file_dir + filename + ".pdf")
    plt.show()
    plt.close()

  def plotCorrelationDiff(self, xlabel="", ylabel="", title="", filename="correlationsDiff", annotation=True):
    """Plotting diff of correlations between columns with numerical data in two
    dataframes and saving result as PDF-file.
    
    :param xlabel: label på x-aksen
    :type xlabel: str
    :param ylabel: label på y-aksen
    :type ylabel: str
    :param title: plottittel
    :type title: str
    :param filename: name of plot file
    :type filename: str

    """

    df1       = self.df1
    df2       = self.df2
    file_dir  = self.file_dir

    corr1 = df1.corr()
    corr2 = df2.corr()
    diff  = corr1 - corr2

    mask = np.zeros_like( diff, dtype=np.bool )
    mask[ np.triu_indices_from( mask ) ] = True
    cmap = sns.diverging_palette( 240, 10, as_cmap=True )

    plt.figure( figsize=( 10, 10 ) )
    ax0 = sns.heatmap( diff, mask=mask, cmap=cmap, vmin=-1
                    , vmax=1, square=True, linewidths=.5, cbar=1
                    , cbar_kws={ "fraction": .05, "shrink": .5, "orientation": "vertical" }
                    , annot=annotation )
    ax0.set_title(title)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(file_dir + filename + ".pdf")
    plt.show()
    plt.close()

  def plotDistributionsOfVariableNumericalVariables(self
                                                  , variable
                                                  , ax1=None
                                                  , ax2=None):
    """Helper function for :class:`plotDistributionsOfVariable` plotting
    the numerical variables in dataframe.

    .. note::
        When plotting a canvas with all histograms ``ax2`` should not be used
        due to layout problems.

    .. todo::
        Legge inn boxplot av hellinger et.al.

    :param variable: name of variable plotting histogram for
    :type variable: str
    :param ax1: axis-object for main-histogram for the variable
    :type ax1: obj
    :param ax2: axis-object for error-histogram placed below main histogram
    :type ax2: obj

    """

    df1       = self.df1
    df2       = self.df2
    name1     = self.name1
    name2     = self.name2
    distributions = self.distributions
    colors    = self.colors
    width_ = abs( max( df1[variable].max(), df2[variable].max() ) - min( df1[variable].min(), df2[variable].min() ) ) / len(distributions[name1][variable][1])
    
    # TODO Flytte feilberegning inn i findDistributions()?
    # df1_err   =  1/np.sqrt(n_samples1[variable]) * np.sqrt( distributions[name1][variable][0] )
    # df2_err   =  1/np.sqrt(n_samples2[variable]) * np.sqrt( distributions[name2][variable][0] )
    df1_err   = distributions[name1 + '_err'][variable]
    df2_err   = distributions[name2 + '_err'][variable]
    ratio     = distributions[name2][variable][0] / distributions[name1][variable][0]
    ratio_err = np.sqrt( (df1_err/distributions[name1][variable][0])**2 \
                       + (df2_err/distributions[name2][variable][0])**2 ) * ratio

    ax1.bar(distributions[name1][variable][1][:-1], distributions[name1][variable][0]
            , align='edge', width=width_, fill=False, edgecolor=colors[0], linewidth=1.3, label=name1, yerr=df1_err)
    ax1.bar(distributions[name2][variable][1][:-1], distributions[name2][variable][0]
            , align='edge', width=width_, fill=False, edgecolor=colors[1], linewidth=1.3, label=name2)
    ax1.legend()
    ax1.set_title(variable)
    
    # Adding errors if not plotting canvas
    if ax2:
      ax2.axhline(y=1, color='k', linestyle='-', linewidth=0.7)
      ax2.errorbar(distributions[name2][variable][1][:-1] + width_/2, ratio, fmt='o', yerr=ratio_err)
      ax2.set_ylim(0,2)
      plt.yticks([0,0.5,1,1.5])
      # Hiding xticks on histogram
      plt.setp(ax1.get_xticklabels(), visible=False)

  def plotDistributionsOfVariableCategoricalVariables(self
                                  , variable
                                  , ax1=None
                                  , ax2=None):
    """Helper function for :class:`plotDistributionsOfVariable` plotting
    the categorical variables in dataframe.

    .. note::
        When plotting a canvas with all histograms ``ax2`` should not be used
        due to layout problems.

    :param variable: name of variable plotting histogram for
    :type variable: str
    :param ax1: axis-object for main-histogram for the variable
    :type ax1: obj
    :param ax2: axis-object for error-histogram placed below main histogram
    :type ax2: obj
    
    """
    name1     = self.name1
    name2     = self.name2
    distributions = self.distributions
    colors = self.colors

    df1_err   = distributions[name1 + '_err'][variable]
    df2_err   = distributions[name2 + '_err'][variable]
  
    dist1_label = distributions[name1][variable][1]
    dist1_height = distributions[name1][variable][0]
    dist2_label = distributions[name2][variable][1]
    dist2_height = distributions[name2][variable][0]
    
    plt.xticks(rotation=45, ha='right')
    plot1 = ax1.bar(dist1_label, dist1_height
            , align='center', width=1, fill=False, edgecolor=colors[0], linewidth=1.3, label=name1, yerr=df1_err)
    plot2 = ax1.bar(dist2_label, dist2_height
            , align='center', width=1, fill=False, edgecolor=colors[1], linewidth=1.3, label=name2)
    ax1.legend((plot1, plot2),(name1, name2))
    ax1.set_title(variable)
    
    # Adding errors of main histogram if not plotting canvas
    if ax2:
      ax2.axhline(y=1, color='k', linestyle='-', linewidth=0.7)
      ratio   = dist2_height / dist1_height
      # TODO add error calculation to docstring.
      ratio_err = np.sqrt( (df1_err/dist1_height)**2 + (df2_err/dist2_height)**2 )*ratio
      # ax2.plot(dist2_label, ratio, 'o')
      ax2.errorbar(dist2_label, ratio, fmt='o', yerr=ratio_err)
      ax2.set_ylim(0, 2)
      plt.yticks([0,0.5,1,1.5])
      # Hiding xticks on histogram
      plt.setp(ax1.get_xticklabels(), visible=False)
    
    if not ax2:
      plt.xlabel(str(variable))

  def plotDistributionsOfVariable(self
                                  , variable
                                  , filename_prefix=''):
    """Plotting distributions for numerical and categorical
    variables in dataframes. Plots are saved in :class:`file_dir`. The
    plots include error bars and visualization of deviation from the
    real dataset.

    :param variable: name of variable plotting histogram for
    :type variable: str
    :param filename_prefix: prefix in filename
    :type filename_prefix: str
    
    """
    df1 = self.df1
    
    # Defining layout of figure
    gs  = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2,1])
    fig = plt.figure(0, figsize=(9,6))
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Checking dtype of variable and plotting data
    if variable in self.num_var:
      self.plotDistributionsOfVariableNumericalVariables(variable, ax1=ax1, ax2=ax2)
    elif variable in self.cat_var:
      self.plotDistributionsOfVariableCategoricalVariables(variable, ax1=ax1, ax2=ax2)
    
    plt.xlabel(str(variable))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(self.file_dir + filename_prefix + '_' + str(variable) + '.pdf')
    plt.show()
    plt.close()

    return 0

  def plotCanvas(self, filename_suffix=''):
    """Plotting canvas of histograms for all variables in the two datasets.
    In order to let plots stay readable a canvas will contain maximum nine
    plots, if there are more than nine variables you will get several
    canvases.

    :param filename_suffix: suffix in filename
    :type filename_suffix: str
    """
    df1 = self.df1
    
    # Fetching all numeric and categorical variables and gathering them in one array
    variables_cat = self.cat_var
    variables_num = self.num_var
    variables = np.concatenate((variables_num, variables_cat))

    # Defining layout of canvas
    num_cols = 3
    num_rows = 3
    num_plots_in_canvas = num_cols*num_rows
    num_numerical  = len(variables_num)
    num_categorial = len(variables_cat)
    num_figures = round((num_numerical+num_categorial)/num_plots_in_canvas)

    for canvas_fig in range(0,num_figures):
      gs       = matplotlib.gridspec.GridSpec(num_rows, num_cols)

      # Creating figure instance and looping through batch of variables in
      # dataframe.
      fig = plt.figure(0, figsize=(15,4*num_rows))
      i   = 0
      # Looping through all variables in dataframes
      for variable in variables[num_plots_in_canvas*canvas_fig:num_plots_in_canvas*(canvas_fig+1)]:
        ax = fig.add_subplot(gs[i])
        i += 1
        if variable in variables_num:
          self.plotDistributionsOfVariableNumericalVariables(variable, ax1=ax)
        elif variable in variables_cat:
          self.plotDistributionsOfVariableCategoricalVariables(variable, ax1=ax)
      
      plt.tight_layout()
      if canvas_fig == 0:
        plt.savefig(self.file_dir + 'canvas_{}.pdf'.format(filename_suffix))
      elif canvas_fig > 0:
        plt.savefig(self.file_dir + 'canvas_{}_{}.pdf'.format(filename_suffix, canvas_fig))
      plt.show()
      plt.close()
  
  def plotPairplot(self):
    """Using seaborn to plot a pairplot of numerical variables.

    .. todo::
        Issues with the layout, not all parts are visible.
    """
    import seaborn as sns

    df1 = self.df1
    df2 = self.df2

    # adding dummy categories for labelling real and synth. dataset
    df1.loc[:, 'dummy'] = 'a'
    df2.loc[:, 'dummy'] = 'b'
    df1.dummy = df1.dummy.astype('category')
    df2.dummy = df2.dummy.astype('category')

    # find all numeric columns and group them in groups of four
    numerical_columns = self.num_var
    i = 0 
    col_groups = {}
    _temp = []
    for col in numerical_columns:
      _temp.append(col)
      i += 1
      if i % 4 == 0:
        col_groups[i] = _temp
        _temp  = []
    
    for _, value in col_groups.items():
      value = value + ['dummy'] # adding dummy-column for coloring in pairplot
      # print(value)
      full_df = pd.concat([df1[value], df2[value]])
      sns.pairplot(full_df, hue='dummy', diag_kind='hist')
      plt.tight_layout()
      plt.show()

  def sum_diff(diff_compare, identificators):
    """Returning the sum of the elements in the diff correlations matrices.

    Parameters
    ----------
      diff_compare : dict
        dict with the absolute difference between the correlation matrices.
      identificators : list
        list of strings which indicates what is different from one
        diff to the next.

    Returns
    -------
      diff_sum : dict
        dictionary with the sum of all elements in the correlation matrices
        as floats.
    """

    diff_sum = {}
    for id_ in identificators:
      # Summing two times (two axes in a matrix)
      diff_sum[id_] = np.sum(np.sum(np.abs(diff_compare[id_])))

    return diff_sum

  def plotVariableImportance(self, clf1, clf2, features, file_dir):

    importance_1 = clf1.feature_importances_
    importance_2 = clf2.feature_importances_
    importance_1_normed  = 100.0 * (importance_1 / importance_1.sum())
    importance_2_normed  = 100.0 * (importance_2 / importance_2.sum())
    importance_1_sorted  = np.argsort(importance_1)
    importance_2_sorted  = np.argsort(importance_2)
    # feature_list = np.array(features)
    pos_1 = np.arange(importance_1_sorted.shape[0]) + .5
    pos_2 = np.arange(importance_2_sorted.shape[0]) + .5
    # gs = matplotlib.gridspec.GridSpec(2,1)
    ax_1 = plt.subplot([0, 0])
    ax_2 = plt.subplot([1, 0])
    # fig.subplots_adjust(left=.2)
    ax_1.barh(pos_1, importance_1_normed[importance_1_sorted], align='center')
    ax_2.barh(pos_2, importance_2_normed[importance_2_sorted], align='center')
    ax_1.yticks(pos_1, features[importance_1_sorted],size=18)
    ax_2.yticks(pos_2, features[importance_2_sorted],size=18)
    plt.xlabel('Relative Importance')
    plt.savefig(file_dir + 'variable_importance.pdf')
    return 0

  def plotConfusionMatrix(self, conf_mat1, conf_mat2, name1, name2, file_dir):

    conf_mat_diff = conf_mat1 - conf_mat2

    gs = matplotlib.gridspec.GridSpec(2,2)
    ax_1 = plt.subplot(gs[1, 0])
    ax_2 = plt.subplot(gs[1, 1])
    ax_3 = plt.subplot(gs[0, :])
    self._plotConfusionMatrixFromLogisticRegression(conf_mat1, title=name1, ax=ax_1)
    self._plotConfusionMatrixFromLogisticRegression(conf_mat2, title=name2, ax=ax_2)
    self._plotConfusionMatrixFromLogisticRegression(conf_mat_diff, title=name2 + ' diff', ax=ax_3)
    plt.tight_layout()
    plt.savefig(file_dir + 'confusion_matrix_logistic_regression.png')
    plt.close()

  def _plotConfusionMatrixFromLogisticRegression(self
                                                , conf_mat
                                                , title=''
                                                , ax=None):
    """Plotting method used in :class:`logisticRegressionBenchmark`

    Parameters
    ----------
      conf_mat : array
        confusion matrix output from ``confusion_matrix`` in ``SciPy``
      title : str
        title of subplot
      ax : object
        axis object for plotting

    """

    sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    ax.set_title('{}'.format(title))