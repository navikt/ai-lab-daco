"""
.. module:: daco
   :platform: Unix, macOS
   :synopsis: Tool for comparing two datasets.

.. moduleauthor:: Jon Vegard Sparre
                  Robindra Prabhu

.. todo::
    - sequential vs tabular data (long term)
    - differences (short term)
    - local and global metrics
    - differential privacy (long term)
    - privacy checks
    - mean, variance, ... ( this is available in pandas)
    - plotting/smart plotting, i.e. show only anomalies
    - pull plots based on the output from pd.dataframe.describe
    - allow setting range, density, binning, etc. for each variable manually?
    - set a random seed globally in class
    - t-test -> ttest_ind or ttest_rel?
    - log-axes in plots
    - row by row comparison - for each row in synthetic dataset, find the row in the original
      dataset with highest match and save the number as an attribute to each row in synth.
      dataset
    - Begrense antall plot i canvas til ni(?)
    - ML-modell-sammenlikning: trene på synt. data, test på ekte.
    - ML-modell-sammenlikning: JordonYoonVanDerSchaar arxiv: 1806.11345v1
    - IdentityDisclosure
    - AttributeDisclosure
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scipy, os
from scipy import stats

class daco():
  """ Class for comparing two Pandas dataframes.

  The purpose of this class is to easily compare datasets in different
  settings, e.g. check if a synthetic version of a dataset is good enoug
  for your use.
  """
  def __init__( self, df1, df2, name1='df1', name2='df2', file_dir="plots/"):
    """
    :param df1: dataframe to be compared. Must have header and dtype for all columns.
    :param df2: dataframe to be compared. Must have header and dtype for all columns.
    :type df1: dataframe
    :type df2: dataframe

    :param file_dir: path to directory where plots are saved
    :type file_dir: str
    :param name1: names of dataframes, used as keys in dictionaries containing info about dataframes
    :param name2: names of dataframes, used as keys in dictionaries containing info about dataframes
    :type name1: str
    :type name2: str 
    """
    # Doing some checks of the dataframes
    self._checkDataframes(df1, df2)

    self.df1      = df1
    self.df2      = df2
    self.file_dir = file_dir
    self.name1 = name1
    self.name2 = name2


    # Creating dicts for saving values for different metrics
    self.p_D_chisquare       = {}
    self.bhattacharyya_dis   = {}
    self.hellinger_div       = {}
    self.kullbackleibler_div = {}
    self.ks2_test_val        = {}
    self.wasserstein_val     = {}

    # Setting plotting colors and some parameters
    self.colors = ['tab:green', 'tab:blue']
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['legend.fontsize'] = 12
    
    # Creating dir for saving plots etc.
    if not os.path.exists(file_dir):
      os.mkdir(file_dir)

  def _checkDataframes(self, df1, df2):
    """Checking whether the dataframes provided fullfills the requirements:
    
    - The input should be pandas dataframes
    - Only float or categorical variables
    - Same column names in both frames
    """

    # TODO implement method for checking that categorical columns in frame 1 and 2
    # contains all values

    for df in [df1, df2]:
      if not isinstance( df, type( pd.DataFrame() ) ):
        raise TypeError("One of the daco-inputs is not a Pandas dataframe")

      col_names = df.select_dtypes(include=[np.number, 'category']).columns
      if not set(df.columns) == set(col_names):
        raise TypeError("Your dataframes has other datatypes than numerical and categorical")
      
    if not set(df1.columns) == set(df2.columns):
      raise ValueError("Your dataframes does not contain the same columns")

  def numericalComparing(self):
    """Compare the mean, variance, etc. of the numerical variable between the
    two dataframes
    """
    df1 = self.df1
    df2 = self.df2

    desc1 = df1.describe()
    desc2 = df2.describe()

    desc_compare = {}

    for variable in df1.select_dtypes(include='number').columns:
      value1 = desc1.mean()[variable]
      value2 = desc2.mean()[variable]
      desc_compare['mean_rel_{}'.format(variable)] = value1/value2

    print(desc_compare)

  def findDistributions(self, bins_='sturges', density=True):
    """Find distributions of all variables/columns in dataframes loaded
    into daco and save them in dictionaries.

    :param bins: number of bins in histogram/distribution or how to find the number of bins.
    :type bins: int 
        
    :param density: if True a normalised distribution is returned.
    :type density: bool
    
    :returns: distributions
    :rtype: dict

    """
    df1 = self.df1
    df2 = self.df2
    name1 = self.name1
    name2 = self.name2

    hist1 = {}
    hist2 = {}
    df1_err = {}
    df2_err = {}

    # looping over all columns containing numerical variables
    column_numerical = df1.select_dtypes(include=[np.number]).columns
    for column in column_numerical:
      min_val = min(df1[ column ].min(), df2[ column ].min())
      max_val = max(df1[ column ].max(), df2[ column ].max())
      range_ = (min_val, max_val)
      
      hist1[ str(column) ] = np.histogram( df1[ column ], bins=bins_, range=range_, density=density )
      hist2[ str(column) ] = np.histogram( df2[ column ], bins=hist1[ str(column) ][1], range=range_, density=density )
      # Calculating the error of each bin: err = 1 / sqrt( N ) * sqrt(  n_i / N ), i.e. the weight is w = 1 / N, where N is the total number of samples in the histogram
      df1_err[ str(column) ] = 1/np.sqrt(np.histogram( df1[ column ], bins=bins_, range=range_)[0].sum()) * np.sqrt( hist1[ str(column) ][0] )
      df2_err[ str(column) ] = 1/np.sqrt(np.histogram( df2[ column ], bins=hist1[ str(column) ][1], range=range_ )[0].sum()) * np.sqrt( hist2[ str(column) ][0] )

    # looping over all columns containing categorical variables
    column_categories = df1.select_dtypes(include=['category']).columns
    for column in column_categories:
      value_count1 = df1[column].value_counts(sort=False)
      value_count2 = df2[column].value_counts(sort=False)
      norm_1 = value_count1.sum()
      norm_2 = value_count2.sum()
      hist1[ str(column) ] = [ value_count1.values / norm_1, value_count1.index.categories ]
      hist2[ str(column) ] = [ value_count2.values / norm_2, value_count2.index.categories ]
      # Calculating the error
      df1_err[ str(column) ] =  1/np.sqrt(value_count1.values) * np.sqrt( hist1[ str(column) ][0] )
      df2_err[ str(column) ] =  1/np.sqrt(value_count2.values) * np.sqrt( hist2[ str(column) ][0] )
    
    distributions = {}
    distributions[ name1 ] = hist1
    distributions[ name2 ] = hist2
    distributions[ name1 + '_err' ] = df1_err
    distributions[ name2 + '_err' ] = df2_err
    
    self.distributions = distributions

    return distributions

  def chisquare(self, var1):
    """Method for calculating the chisquare test using scipy.stats.chisquare.

    :param var1: name of variable in the dataset to do the chisquare test.
    :type var1: str

    :returns: D, p
    :rtype: float

    """
    distributions = self.distributions
    p_D_chisquare = self.p_D_chisquare

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    D, p = scipy.stats.chisquare( dist1, dist2 )

    # Saving results in dictionary
    p_D_chisquare[var1] = { 'D': D, 'pvalue': p }

    return D, p

  def hellinger(self, var1):
    """Calculate the Hellinger divergence for the distributions of
    var1 in the two dataframes. 
    See `https://en.wikipedia.org/wiki/Hellinger_distance <https://en.wikipedia.org/wiki/Hellinger_distance>`_

    :param var1: name of variable in the datasets to do calculate the Hellinger divergence for.
    :type var1: str
    
    :returns: hellinger_div. Ouput value is in range `[0, sqrt(2)]`.
    :rtype: float

    """
    distributions = self.distributions
    hellinger_div = self.hellinger_div

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    hellinger_div_value = np.sqrt( np.sum( ( np.sqrt(dist1) - np.sqrt(dist2) )**2 ) )

    hellinger_div[var1] = hellinger_div_value

    return hellinger_div_value

  def kullbackleibler(self, var1):
    """Calculate Kullback-Leibler divergence for the distributions of
    var1 in the two dataframes with scipy.stats.entropy.

    - `https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence <https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence>`_
    - `https://en.wikipedia.org/wiki/Kullback–Leibler_divergence <https://en.wikipedia.org/wiki/Kullback–Leibler_divergence>`_

    :param var1: name of variable to do calculate the Kullback-Leibler divergence for. Must be contained in the dataframes.
    :type var1: str
    
    :returns: kb_div
    :rtype: float

    """
    import scipy.special as spec

    distributions = self.distributions
    kullbackleibler_div = self.kullbackleibler_div

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    # return np.sum( np.where( p != 0., p * np.log( p / np.where( q != 0., q, 1 ) ), 0 ) )
    # return sp.entropy(p, q)

    # kb_div = spec.kl_div(dist1, dist2)
    kb_div = stats.entropy( dist1, dist2 )

    kullbackleibler_div[var1] = kb_div

    return kb_div

  def wasserstein(self, var1):
    """
    """
    from scipy.stats import wasserstein_distance

    distributions   = self.distributions
    wasserstein_val = self.wasserstein_val

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    wasserstein_val_ = wasserstein_distance(dist1, dist2)

    wasserstein_val[var1] = wasserstein_val_

    return wasserstein_val_

  
  def plotDistanceMetrics(self):
    df1 = self.df1

    for column in df1.select_dtypes(include='number').columns:
      self.kullbackleibler(column)
      self.bhattacharyya(column)
      self.hellinger(column)

    # forcing KL to be a number between 1 and 0.
    kl_array = 1 - np.exp(- np.array(list(self.kullbackleibler_div.values())))
    bha_array = np.array(list(self.bhattacharyya_dis.values()))
    hel_array = np.array(list(self.hellinger_div.values())) / np.sqrt(2) # maybe a stupid normalization

    # TODO bytte til seaborn --> må antakeligvis legge arrays over inn i en dataframe
    plt.boxplot([kl_array, bha_array, hel_array], showmeans=True)
    plt.show()

  def bhattacharyya(self, var1):
    """Calculate the Bhattacharyya distance for the distributions of
    var1 in the two dataframes.
    See `https://en.wikipedia.org/wiki/Bhattacharyya_distance <https://en.wikipedia.org/wiki/Bhattacharyya_distance>`_

    :param var1: name of variable in the datasets to do calculate the Bhattacharyya distance for.
    :type var1: str
    
    :returns: b_dis
    :rtype: float

    """

    distributions = self.distributions
    bhattacharyya_dis = self.bhattacharyya_dis

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    def normalize(h):
      return h/np.sum(h)

    b_dis = 1 - np.sum( np.sqrt( np.multiply( normalize(dist1), normalize(dist2) ) ) )

    bhattacharyya_dis[var1] = b_dis

    return b_dis

  def ks2_test(self, var):
    """Method using the scipy.stats.ks_2samp for computing the Kolmogorov-
    Smirnov statistic on two samples. The result is added to a dictionary
    :class:`ks2_test_val` keeping the results of all calculations.

    Parameters
    ----------
    var1 : str
      name of variable in the datasets to do the test on.

    Returns
    -------
    statistic : float
      KS statistic
    pvalue : float
      two-tailed p-value

    """
    df1 = self.df1
    df2 = self.df2

    assert var in set(df1.select_dtypes(include='number').columns), "'{}' is invalid, it must be a continuous variable.".format(var)

    data1 = df1[var].values
    data2 = df2[var].values

    statistic, pvalue = stats.ks_2samp(data1, data2) 

    self.ks2_test_val[var] = {'statistic': statistic, 'pvalue': pvalue }

    return statistic, pvalue

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
        When plotting a canvas with all histograms `ax2` should not be used
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
        When plotting a canvas with all histograms ax2 should not be used
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
    variables in dataframes. Plots are saved in self.file_dir. The
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

  def logisticRegressionBenchmark(self, target=[], features=[], test_size=0.2, eval_size=0.2):
    """Method for training a logistic regression-model on the datasets
    and investigate the differences in the model and predictions. The
    results and models are saved as class variables.

    Main features:

    - Training LR-models on synth. and real data (and save them in this class)
    - Predicting *N* samples
    - Comparing the accuracy of the two models (several measures possible)
    - Confusion matrix + classification_report from sklearn
    - Feature importance

    :param target: target values
    :type target: list
    :param features: features to use in training/predictions
    :type features: list
    :param test_size: size of test set
    :type test_size: float
    :param eval_size: size of evaluation set
    :type eval_size: float

    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    X_train1, X_val1, y_train1, y_val1, X_test1, y_test1 = self.dataPrep(target, features, test_size, eval_size)
    X_train2, X_val2, y_train2, y_val2, X_test2, y_test2 = self.dataPrep(target, features, test_size, eval_size)

    # Training models and calculating their accuracies
    clf1 = LogisticRegression().fit(X_train1, y_train1)
    s1 = clf1.score(X_test1, y_test1)

    clf2 = LogisticRegression().fit(X_train2, y_train2)
    s2 = clf2.score(X_test2, y_test2)

    # saving models in class
    self.LR_model1  = clf1
    self.LR_model2  = clf2
    self.score_clf1 = s1
    self.score_clf2 = s2

    name1    = self.name1
    name2    = self.name2
    file_dir = self.file_dir

    # Evaulating and finding confusion matrices
    predictions1  = clf1.predict(X_val1)
    predictions2  = clf2.predict(X_val2)
    conf_mat1     = confusion_matrix(y_true=y_val1, y_pred=predictions1)
    conf_mat2     = confusion_matrix(y_true=y_val2, y_pred=predictions2)
    conf_mat_diff = conf_mat1 - conf_mat2

    # Plotting confusion matrices
    gs = matplotlib.gridspec.GridSpec(2,2)
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1])
    ax3 = plt.subplot(gs[0, :])
    self._plotConfusionMatrixFromLogisticRegression(conf_mat1, title=name1, ax=ax1)
    self._plotConfusionMatrixFromLogisticRegression(conf_mat2, title=name2, ax=ax2)
    self._plotConfusionMatrixFromLogisticRegression(conf_mat_diff, title=name2 + ' diff', ax=ax3)
    plt.tight_layout()
    plt.savefig(file_dir + 'confusion_matrix_logistic_regression.png')
    plt.show()
    plt.close()

  def dataPrep(self, target, features, test_size, eval_size):
    """Data preparation for the ML-models used in DACO. Takes in the two dataframes
    applies one hot encoding on categorical variables, and splits them into train,
    test, and evaluation sets.

    :param target: target values
    :type target: list
    :param features: features to use in training/predictions
    :type features: list
    :param test_size: size of test set
    :type test_size: float
    :param eval_size: size of evaluation set
    :type eval_size: float
    """
    from sklearn.model_selection import train_test_split

    # One hot encoding categorical values
    # TODO if not all values are present in both df1 and df2 we will get
    # different columns in each dataframe, must be fixed
    oneHotEncode = lambda df: pd.get_dummies(df, columns=df[features].select_dtypes(include='category').columns)

    df1 = self.df1
    df2 = self.df2

    df1 = oneHotEncode(df1)
    df2 = oneHotEncode(df2)

    # generating new features list with one hot encoded features
    features_new = []
    for column in features:
      for df_col in df1.columns:
        if df_col.startswith(column):
          features_new.append(df_col)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(df1[features_new]
                                                          , df1[target]
                                                          , test_size=test_size)
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train1, y_train1
                                                      , test_size=eval_size)

    return X_train1, X_val1, y_train1, y_val1, X_test1, y_test1

  def _plotConfusionMatrixFromLogisticRegression(self
      , conf_mat
      , title=''
      , ax=None):
    """Plotting method used in :class:`logisticRegressionBenchmark`

    Parameters
    ----------
      conf_mat : array
        confusion matrix output from `confusion_matrix` in `scipy`
      title : str
        title of subplot
      ax : object
        axis object for plotting

    """

    sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    ax.set_title('{}'.format(title))
  
  def plotPairplot(self):
    """Using seaborn to plot a pairplot
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
    
    for key, value in col_groups.items():
      value = value + ['dummy'] # adding dummy-column for coloring in pairplot
      print(value)
      full_df = pd.concat([df1[value], df2[value]])
      sns.pairplot(full_df, hue='dummy', diag_kind='hist')
      plt.show()

  def rowMatching(self, atol_=1e-1, rtol_=1e-1):
    """Method which loops over each row in synthetic dataset and finds the row
    in the original dataset with highest match in percent, and saves this number
    as an attribute to each row in the synthetic dataset.
    Using numpy.isclose as a matching measure for numerical values.

    Parameters
    ----------
    atol : float
      The relative tolerance parameter, see Numpy docs.
    rtol : float
      The absolute tolerance parameter, see Numpy docs.

    Returns
    -------
    """
    # fetching only columns with numerical values
    df1 = self.df1.select_dtypes(include=[np.number])
    df2 = self.df2.select_dtypes(include=[np.number])
    
    # initializing dict that will be filled with entries on
    # the form <index in df2> : [<index in df1>, match_value]
    match_values = np.empty((len(df2), 3))
    
    len_row = len(df1.columns)
    
    i = 0
    for row2 in df2.itertuples():
      row_match = [None, 0]
      for row1 in df1.itertuples():
        # applying np.isclose and counting number of elements inside our tolerances
        match = np.isclose(row1[1:], row2[1:], atol=atol_, rtol=rtol_)
        match_rel = match.sum() / len_row
        if match_rel >= row_match[1]:
          row_match = [row1[0], match_rel]
      match_values[i] = np.array([row2[0], row_match[0], row_match[1]])
      i += 1

    self.match_values = match_values
