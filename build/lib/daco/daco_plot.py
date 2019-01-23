# -*- coding: utf-8 -*-

"""
.. module:: daco_plot
   :platform: Unix, macOS
   :synopsis: Plotting module for DACO.

.. moduleauthor:: Jon Vegard Sparre
                  Robindra Prabhu

.. todo::
    - plotting/smart plotting, i.e. show only anomalies
    - pull plots based on the output from pd.dataframe.describe
    - allow setting range, density, binning, etc. for each variable manually?
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scipy, os
from scipy import stats

class plot():
  """ Class for plotting the metrics found in :class:``daco_main``.
  """
  def plotDistanceMetrics(self):
    """Plot boxplot of the distance metrics Kullback-Leibler, Bhattacharyya, and
    Hellinger for all numerical variables.
    """
    kl_array, bha_array, hel_array = self._findAndNormalizeDistances()

    # TODO bytte til seaborn --> må antakeligvis legge arrays over inn i en dataframe
    plt.boxplot([kl_array, bha_array, hel_array], showmeans=True)
    plt.show()

  def plotCorrelation(self, xlabel="", ylabel="", title="", filename="correlations"):
    """Plotting correlations between columns in a dataframe and saving as PNG-file.
    
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
                    , annot=True )
    ax0.set_title(title)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    plt.savefig(file_dir + filename + ".png")
    plt.show()
    plt.close()

  def plotCorrelationDiff(self, xlabel="", ylabel="", title="", filename="correlations"):
    """Plotting diff of correlations between columns in two dataframes and saving
    result as PNG-file.
    
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
                    , annot=True )
    ax0.set_title(title)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    plt.savefig(file_dir + filename + ".png")
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
    width_ = abs( max( df1[variable].max(), df2[variable].max() ) - min( df1[variable].min(), df2[variable].min() ) ) / len(distributions['df1'][variable][1])
    
    # TODO Flytte feilberegning inn i findDistributions()?
    # df1_err   =  1/np.sqrt(n_samples1[variable]) * np.sqrt( distributions['df1'][variable][0] )
    # df2_err   =  1/np.sqrt(n_samples2[variable]) * np.sqrt( distributions['df2'][variable][0] )
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
      ax2.errorbar(distributions['df2'][variable][1][:-1] + width_/2, ratio, fmt='o', yerr=ratio_err)
      ax2.set_ylim(0,2)
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

    plt.xticks(rotation=45, ha='right')
    ax1.bar(distributions['df1'][variable][1], distributions['df1'][variable][0]
            , align='center', width=1, fill=False, edgecolor=colors[0], linewidth=1.3, label=name1,yerr=df1_err)
    ax1.bar(distributions['df2'][variable][1], distributions['df2'][variable][0]
            , align='center', width=1, fill=False, edgecolor=colors[1], linewidth=1.3, label=name2)
    ax1.legend()
    ax1.set_title(variable)
    
    # Adding errors of main histogram if not plotting canvas
    if ax2:
      ratio   = distributions['df2'][variable][0] / distributions['df1'][variable][0]
      ratio_err = np.sqrt( (df1_err/distributions['df1'][variable][0])**2 + (df2_err/distributions['df2'][variable][0])**2 )*ratio
      ax2.axhline(y=1, color='k', linestyle='-', linewidth=0.7)
      # ax2.plot(distributions['df2'][variable][1], ratio, 'o')
      ax2.errorbar(distributions['df2'][variable][1], ratio, fmt='o', yerr=ratio_err)
      ax2.set_ylim(0,2)
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
    fig = plt.figure(0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Checking dtype of variable and plotting data
    if variable in df1.select_dtypes(include=[np.number]).columns:
      self.plotDistributionsOfVariableNumericalVariables(variable, ax1=ax1, ax2=ax2)
    elif variable in df1.select_dtypes(include='category').columns:
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
    variables_num = df1.select_dtypes(include=[np.number]).columns
    variables_cat = df1.select_dtypes(include='category').columns
    variables = np.concatenate((variables_num, variables_cat))

    # Defining layout of canvas
    num_plots_in_canvas = 9
    num_numerical  = len(variables_num)
    num_categorial = len(variables_cat)
    num_figures = round((num_numerical+num_categorial)/num_plots_in_canvas)

    for canvas_fig in range(0,num_figures):
      num_cols = 3
      num_rows = 3
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
      plt.savefig(self.file_dir + 'canvas_' + filename_suffix + '_{}_.pdf'.format(canvas_fig))
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
    columns = df1.select_dtypes(include=[np.number]).columns
    i = 0 
    col_groups = {}
    _temp = []
    for col in columns:
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
