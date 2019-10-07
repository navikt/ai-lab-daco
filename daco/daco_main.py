#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module:: daco
   :platform: Unix, macOS
   :synopsis: Tool for comparing two datasets.

.. moduleauthor:: Jon Vegard Sparre
                  Robindra Prabhu

.. todo::
    - add support for finding Kulback et.al. for categorical data
    - sequential vs tabular data (long term)
    - local and global metrics
    - differential privacy (long term)
    - privacy checks
    - mean, variance, ... ( this is available in pandas)
    - plotting/smart plotting, i.e. show only anomalies
    - pull plots based on the output from pd.dataframe.describe
    - allow setting range, density, binning, etc. for each variable manually?
    - set a random seed globally in class
    - log-axes in plots
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
from daco.daco_plot import plot

class daco(plot):
  """ Class for comparing two Pandas dataframes.

  The purpose of this class is to easily compare datasets in different
  settings, e.g. check if a synthetic version of a dataset is good enoug
  for your use.
  """
  def __init__(self, df1, df2, name1='df1', name2='df2', file_dir="plots/"):
    """
    :param df1: dataframe to be compared. Must have header and dtype for all columns.
    :param df2: dataframe to be compared. Must have header and dtype for all columns. This is\
    treated as the synthetic version of df1 throughout the module.
    :type df1: dataframe
    :type df2: dataframe

    :param file_dir: path to directory where plots are saved
    :type file_dir: str
    :param name1: names of dataframes, used as keys in dictionaries containing info about dataframes
    :param name2: names of dataframes, used as keys in dictionaries containing info about dataframes
    :type name1: str
    :type name2: str
    """

    self._checkDataframes(df1, df2)

    self.df1      = df1
    self.df2      = df2
    self.file_dir = file_dir
    self.name1 = name1
    self.name2 = name2


    # Creating dicts for saving different metrics
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
      if not isinstance(df, type(pd.DataFrame())):
        raise TypeError("One of the daco-inputs is not a Pandas dataframe")

      col_names = df.select_dtypes(include=[np.number, 'category']).columns
      if not set(df.columns) == set(col_names):
        raise TypeError("Your dataframes has other datatypes than numerical and categorical")

    if not set(df1.columns) == set(df2.columns):
      raise ValueError("Your dataframes does not contain the same columns")

  def numericalComparing(self):
    """Compare the mean, variance, etc. of the numerical variable between the
    two dataframes. For each numerical variable we calculate the mean of the
    values in both dataframes, then we divide the mean value of the variable in
    the first dataframe on the second one to find how much they differ.

    Returns
    -------
      desc_compare : dict
        contains the relative mean values of all numerical variables in both
        dataframes.

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
    return(desc_compare)

  def findDistributions(self, bins_='sturges'):
    r"""Find normalized distributions of all variables/columns in dataframes
    loaded into daco and save them in dictionaries.

    The error is calculated as
    
    .. math::
      \epsilon = \frac{1}{\sqrt{N}} \sqrt{\frac{n_i}{N}},

    where :math:`N` is the number of data samples and :math:`n_i` is the number of samples in
    the bin.

    Parameters
    ----------
      bins : int
        number of bins in histogram/distribution or how to find the number of bins.
      density :bool
        if True a normalised distribution is returned.

    Returns
    -------  
      distributions : dict
        All normalized distributions gathered in a dict.

    Examples
    --------
    By giving two Pandas dataframes we find the distributions:

    >>> daco_obj = daco(df1,df2)
    >>> dist = daco_obj.findDistributions()
    >>> print(dist)
    {'df1': {'age': (array([0.01964384, 0.0258, ...]}, 'df2': {...}}

    """
    df1 = self.df1
    df2 = self.df2
    name1 = self.name1
    name2 = self.name2

    hist1 = {}
    hist2 = {}
    df1_err = {}
    df2_err = {}
    #
    # TODO Sjekke ut error-bars på numeriske plott....
    #
    # looping over all columns containing numerical variables
    column_numerical = df1.select_dtypes(include=[np.number]).columns
    for column in column_numerical:
      x1 = df1[column]
      x2 = df2[column]
      min_val = min(x1.min(), x2.min())
      max_val = max(x1.max(), x2.max())
      range_ = (min_val, max_val)
      hist1[str(column)] = np.histogram(x1
                                        , bins=bins_
                                        , range=range_)
      hist2[str(column)] = np.histogram(x2
                                        , bins=hist1[str(column)][1]
                                        , range=range_)
        
      for x, hist, df_err in [(x1, hist1, df1_err), (x2, hist2, df2_err)]:
        # Calculating the error of each bin: err = 1 / sqrt(N) * sqrt(n_i / N) = sqrt(n_i) / N,
        # i.e. the weight is w = 1 / N, where N is the total number of samples in the histogram
        df_err[str(column)] = 1 / np.histogram(x
                                          , bins=hist1[str(column)][1] # use same binning as above
                                          , range=range_)[0].sum() * np.sqrt(hist[str(column)][0])

        # Normalizing histogram
        # TODO lag en funksjon av dette
        a = hist[str(column)][0] / len(x)
        hist[str(column)] = (a, hist[str(column)][1])
        
    # looping over all columns containing categorical variables
    column_categories = df1.select_dtypes(include=['category']).columns
    for column in column_categories:
      x1 = df1[column]
      x2 = df2[column]
      for x, hist, df_err in [(x1, hist1, df1_err), (x2, hist2, df2_err)]:
        # Counting values and normalizing for each category.
        value_count = x.value_counts(sort=False)
        norm = value_count.sum()
        hist[str(column)] = [value_count.values / norm, value_count.index.categories]
        # Calculating the error
        df_err[str(column)] = 1 / np.sqrt(value_count.values) * np.sqrt(hist[str(column)][0])

    distributions = {}
    for name, hist, df_err in [(name1, hist1, df1_err), (name2, hist2, df2_err)]:
      distributions[name] = hist
      distributions[name + '_err'] = df_err

    self.distributions = distributions

    return distributions

  def findDistributionOfNumericalVariable(self, var, range_, bins_):
    """Calculate distribution of a single variable.
    """
    df1 = self.df1
    df2 = self.df2
    name1 = self.name1
    name2 = self.name2
      
    x1 = df1[var]
    x2 = df2[var]
    
    hist1 = {}
    hist2 = {}
    df1_err = {}
    df2_err = {}
      
    hist1[str(var)] = np.histogram(x1
                                      , bins=bins_
                                      , range=range_)
    hist2[str(var)] = np.histogram(x2
                                      , bins=hist1[str(var)][1]
                                      , range=range_)
        
    for x, hist, df_err in [(x1, hist1, df1_err), (x2, hist2, df2_err)]:
      # Calculating the error of each bin: err = 1 / sqrt(N) * sqrt(n_i / N) = sqrt(n_i) / N,
      # i.e. the weight is w = 1 / N, where N is the total number of samples in the histogram
      df_err[str(var)] = 1 / np.histogram(x
                                             , bins=hist1[str(var)][1] # use same binning as above
                                             , range=range_)[0].sum() * np.sqrt(hist[str(var)][0])
      # Normalizing histogram
      a = hist[str(var)][0] / len(x)
      hist[str(var)] = (a, hist[str(var)][1])

    dist = {}
    for name, hist, df_err in [(name1, hist1, df1_err), (name2, hist2, df2_err)]:
      dist[name] = hist
      dist[name + '_err'] = df_err

    return dist

  def attributeDisclosure(self, var, other_vars=[]):
    """Method doing an attribute disclosure-calculation.

    Idea: User gives a list with variables which an intruder may want to learn, this
    method will use the remaining variables to create subsets of the synthetic dataset
    w.r.t. each person in original dataset and look at how equal the sensitive variables
    are.
    """

    df1 = self.df1
    df2 = self.df2

    if len(other_vars) == 0:
      other_vars = set(var).symmetric_difference(set(df1.columns))
    
    count_matches = pd.DataFrame(columns=['index', 'matches', 'n_values', 'true_matches'])

    for index, person in df1.iterrows(): 
      person_values = person[other_vars].values 
      subset_idx = (df2[other_vars] == person_values).all(axis=1)
      subset = df2[subset_idx] # subset = ekvivalensklasse
      n_values = subset[var].nunique() # l-diversity
      true_matches = (subset[var] == person[var]).sum() / subset.shape[0] # Andel i subset som også matcher på sensitiv variabel.
      count_matches = count_matches.append([{'index': index, 'matches': subset.shape[0], 'n_values': n_values, 'true_matches': true_matches}])

    self.matches = count_matches
    self.disclosure_risks = {}
    self.disclosure_risks['true_match_max'] = count_matches.true_matches.max() # 
    self.disclosure_risks['true_match_mean'] = count_matches.true_matches.mean()
    self.disclosure_risks['true_match_median'] = count_matches.true_matches.median()
    self.disclosure_risks['true_match_data'] = count_matches.true_matches


  def chisquare(self, var1):
    """Method for calculating the chisquare test using scipy.stats.chisquare.

    Parameters
    ----------
      var1 : str
        name of variable in the dataset to do the chisquare test.

    Returns
    -------
      D : float
        The chi-squared test statistic
      p : float
        p-value

    """
    distributions = self.distributions
    p_D_chisquare = self.p_D_chisquare

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    D, p = scipy.stats.chisquare(dist1, dist2)

    # Saving results in dictionary
    p_D_chisquare[var1] = {'D': D, 'pvalue': p}

    return D, p

  def _hellingerDivergence(self, dist1, dist2):
    """Private method calculating the Hellinger divergence.
    """

    hellinger_div_value = np.sqrt(np.sum((np.sqrt(dist1) - np.sqrt(dist2))**2))

    return hellinger_div_value

  def hellinger(self, var1):
    """Calculate the Hellinger divergence for the distributions of
    var1 in the two dataframes.
    See `https://en.wikipedia.org/wiki/Hellinger_distance \
    <https://en.wikipedia.org/wiki/Hellinger_distance>`_

    :param var1: name of variable in the datasets to do calculate the Hellinger divergence for.
    :type var1: str

    :returns: hellinger_div. Ouput value is in range ``[0, sqrt(2)]``.
    :rtype: float

    """
    distributions = self.distributions
    hellinger_div = self.hellinger_div

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    hellinger_div_value = self._hellingerDivergence(dist1, dist2)

    hellinger_div[var1] = hellinger_div_value

    return hellinger_div_value

  def kullbackleibler(self, var1):
    """Calculate Kullback-Leibler divergence for the distributions of
    var1 in the two dataframes with scipy.stats.entropy.

    - `https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence \
    <https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence>`_
    - `https://en.wikipedia.org/wiki/Kullback–Leibler_divergence \
    <https://en.wikipedia.org/wiki/Kullback–Leibler_divergence>`_

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
    kb_div = stats.entropy(dist1, dist2)

    kullbackleibler_div[var1] = kb_div

    return kb_div

  def wasserstein(self, var1):
    """Calculating the Wasserstein/"earth mover's distance" with ``SciPy``.

    Parameters
    ----------
      var1 : str
         name of variable to calculate for
    """
    from scipy.stats import wasserstein_distance

    distributions   = self.distributions
    wasserstein_val = self.wasserstein_val

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    wasserstein_val_ = wasserstein_distance(dist1, dist2)

    wasserstein_val[var1] = wasserstein_val_

    return wasserstein_val_

  def bhattacharyya(self, var1):
    """Calculate the Bhattacharyya distance for the distributions of
    var1 in the two dataframes.
    See `https://en.wikipedia.org/wiki/Bhattacharyya_distance \
    <https://en.wikipedia.org/wiki/Bhattacharyya_distance>`_

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

    b_dis = 1 - np.sum(np.sqrt(np.multiply(normalize(dist1), normalize(dist2))))

    bhattacharyya_dis[var1] = b_dis

    return b_dis

  def _findAndNormalizeDistances(self):
    """Calculate and normalize distances for all numerical variables.

    Returns
    -------
      kl_array : array
        Kullback-Leibler values
      bha_array : array
        Bhattacharyya values
      hel_array : array
        Hellinger values
    """
    df1 = self.df1

    for column in df1.select_dtypes(include='number').columns:
      self.kullbackleibler(column)
      self.bhattacharyya(column)
      self.hellinger(column)

    # forcing KL to be a number between 1 and 0.
    kl_array = 1 - np.exp(-np.array(list(self.kullbackleibler_div.values())))
    bha_array = np.array(list(self.bhattacharyya_dis.values()))
    hel_array = np.array(list(self.hellinger_div.values())) / np.sqrt(2) # maybe a stupid normalization

    return kl_array, bha_array, hel_array

  def printDistances(self):
    """Print a nice markdown table with the distance metrics for all numerical
    variables.

    Returns
    -------
      distance_values : dict
        a dict containing all values calculated
    """
    df1 = self.df1

    kl_array, bha_array, hel_array = self._findAndNormalizeDistances()

    print("| Variable             | Kullback | Bhattacharyya | Hellinger |")
    for column, kl, bha, hel in zip(df1.select_dtypes(include='number').columns, kl_array, bha_array, hel_array):
      print("| {:20} | {:8.2f} | {:13.2f} | {:9.2f} |".format(column, kl, bha, hel))

    return 0

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

  def logisticRegressionBenchmark(self, target, features, test_size=0.2, eval_size=0.2):
    """Method for training a logistic regression-model on the datasets
    and investigate the differences in the model and predictions. The
    results and models are saved as class variables.

    The following is done by this function:

    - Training LR-models on synth. and real data (and save them in this class)
    - Predict *N* samples
    - Comparing the accuracy of the two models (several measures possible)
    - Plotting confusion matrix

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
    from sklearn.metrics import confusion_matrix, classification_report

    name1    = self.name1
    name2    = self.name2
    file_dir = self.file_dir

    X_train1, X_val1, y_train1, y_val1, X_test1, y_test1 = self.dataPrep(target, features, test_size, eval_size, name1)
    X_train2, X_val2, y_train2, y_val2, X_test2, y_test2 = self.dataPrep(target, features, test_size, eval_size, name2)

    # Training models and calculating their accuracies
    clf1 = LogisticRegression().fit(X_train1, y_train1)
    s1 = clf1.score(X_test1, y_test1)

    clf2 = LogisticRegression().fit(X_train2, y_train2)
    s2 = clf2.score(X_test2, y_test2)

    # saving models in object
    self.LR_model1  = clf1
    self.LR_model2  = clf2
    self.score_clf1 = s1
    self.score_clf2 = s2

    # Evaulating and calculating confusion matrices
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

    # Plotting variable importance
    importance_1 = clf1.feature_importances_
    importance_2 = clf2.feature_importances_
    importance_1_normed  = 100.0 * (importance_1 / importance_1.sum())
    importance_2_normed  = 100.0 * (importance_2 / importance_2.sum())
    importance_1_sorted  = np.argsort(importance_1)
    importance_2_sorted  = np.argsort(importance_2)
    feature_list = np.array(features)
    pos_1 = np.arange(importance_1_sorted.shape[0]) + .5
    pos_2 = np.arange(importance_2_sorted.shape[0]) + .5
    gs = matplotlib.gridspec.GridSpec(2,1)
    ax_1 = plt.subplot([0, 0])
    ax_1 = plt.subplot([1, 0])
    # fig.subplots_adjust(left=.2)
    ax1.barh(pos_1, importance_1_normed[importance_1_sorted], align='center')
    ax1.barh(pos_2, importance_2_normed[importance_2_sorted], align='center')
    ax1.yticks(pos_1, features[importance_1_sorted],size=18)
    ax2.yticks(pos_2, features[importance_2_sorted],size=18)
    plt.xlabel('Relative Importance')
    plt.savefig(file_dir + 'variable_importance.pdf')
    return 0

  def dataPrep(self, target, features, test_size, eval_size, name):
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
    :param name: name of dataset to prepare
    :type name: str
    """
    from sklearn.model_selection import train_test_split

    # One hot encoding categorical values
    # TODO if not all values are present in both df1 and df2 we will get
    # different columns in each dataframe, must be fixed
    oneHotEncode = lambda df: pd.get_dummies(df, columns=df[features].select_dtypes(include='category').columns)

    if name == self.name1:
      df = self.df1
      df = oneHotEncode(df)
    elif name == self.name2:
      df = self.df2
      df = oneHotEncode(df)

    # generating new features list with one hot encoded features
    features_new = []
    for column in features:
      for df_col in df.columns:
        if df_col.startswith(column):
          features_new.append(df_col)

    _X_train, X_test, _y_train, y_test = train_test_split(df[features_new]
                                                          , df[target]
                                                          , test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(_X_train, _y_train
                                                      , test_size=eval_size)

    return X_train, X_val, y_train, y_val, X_test, y_test

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
      match_values : array
        array with dimensions (len(df2), 3) where each row contains
        ``[<index in df2>, <index in df1>, match_value]``.
    """
    # fetching columns with numerical values
    df1 = self.df1.select_dtypes(include=[np.number])
    df2 = self.df2.select_dtypes(include=[np.number])
    
    # initializing dict that will be filled with entries on
    # the form [<index in df2>, <index in df1>, match_value]
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

    return match_values

  def hellingerRowForRow(self):
    """Method for comparing finding the pair of rows in the datasets that have
    the best match.

    .. warning::
        Very slow implementation!

    .. todo::
        Parallellize?
    """

    # fetching columns with numerical values
    df1 = self.df1.select_dtypes(include=[np.number])
    df2 = self.df2.select_dtypes(include=[np.number])

    # initializing dict that will be filled with entries on
    # the form [<index in df2>, <index in df1>, match_value]
    match_values = np.empty((len(df2), 3))
    
    len_row = len(df1.columns)
    
    i = 0
    for row2 in df2.itertuples():
      row_match = [None, 0]
      for row1 in df1.itertuples():
        # applying np.isclose and counting number of elements inside our tolerances
        match = self._hellingerDivergence(row1[1:], row2[1:])
        match_rel = match.sum() / len_row
        if match_rel >= row_match[1]:
          row_match = [row1[0], match_rel]
      match_values[i] = np.array([row2[0], row_match[0], row_match[1]])
      i += 1

    self.hellinger_row = match_values
    return match_values

  def syntheticRankingAgreement(self, model_scores=None, target=None, features=None):
    r"""Method for checking whether the synthetic dataset is useful in machine learning
    contexts. We use the "Synthetic Ranking Agreement"-method, SRA for short,
    see `https://arxiv.org/pdf/1806.11345v1.pdf <https://arxiv.org/pdf/1806.11345v1.pdf>`_
    It is defined as 
    
    .. math::
      \text{SRA} = \frac{1}{k(k-1)} \sum^{k}_{i=1}\sum_{j\neq i} \mathbb{I}\left( (R_i - R_j) \times (S_i - S_j) \right)

    where :math:`R_k` and :math:`S_k` represents the performance score of algorithm :math:`k` on
    the real dataset :math:`R` and the synthetic dataset :math:`S` respectively. The SRA can be
    thought of as the (emprical) probability of a comparison on the synthetic data beain "correct".

    Parameters
    ----------
      model_scores : dict
        dict with scores for the different algorithms on the real and synthetic dataset.
      target : list
        list of training targets (if not ``model_scores`` is given)
      features : list
        list of training features (if not ``model_scores`` is given)
      
    Returns
    -------
      sra : float
        a single value which is the result of the formula given above.
    """
    name1 = self.name1
    name2 = self.name2
    
    if model_scores is None:
      _, model_scores = self.trainAndTestModels(target, features)
      K = len(model_scores[name1]) # number of models/algorithms
      S = model_scores[name1] # synthetic model scores
      R = model_scores[name2] # real model scores
    elif model_scores is not None:
      K = len(model_scores[name1])
      S = model_scores[name1]
      R = model_scores[name2]

    sra = 0
    for i in range(0,K):
      for j in range(0,K):
        if j != i:
          sra += ((R[i] - R[j])*(S[i] - S[j])) > 0
    sra *= 1/(K*(K-1))

    return sra

  def trainAndTestModels(self
                        , target
                        , features
                        , models=None
                        , test_size=0.2
                        , eval_size=0.2):
    """Method for training several ML-models on the two datasets given to this
    class and do testing as specified by the user. The models and their test
    scores are saved in dictionaries.
    
    Parameters
    ----------
      target : list
        target values
      features: list
        features to use in training/predictions
      models : list
        list of tuples with models and a dict with parameters
      test_size : float
        size of test set
      eval_size : float
        size of evaluation set
      
    Returns
    -------
      model_dict : dict
        dictionary with the models trained
      model_scores : dict
        dictionary with test scores for the models trained
    """
    name1 = self.name1
    name2 = self.name2
    
    model_dict = {}
    model_scores = {}

    for name in (name1, name2):
      X_train, _, y_train, _, X_test, y_test = self.dataPrep(target
                                                            , features
                                                            , test_size
                                                            , eval_size
                                                            , name)

      data = (X_train, y_train, X_test, y_test)
      model_dict_, model_scores_ = self._trainSeveralModels(name, data, models)

      model_dict.update(model_dict_)
      model_scores.update(model_scores_)

    self.model_dict = model_dict
    self.model_scores = model_scores

    return model_dict, model_scores

  def _trainSeveralModels(self, name, data, models=None):
    """Private method for :class:`trainSynthTestReal` and :class:`syntheticRankingAgreement`.
    Models and their scores are saved in dictionaries.
    
    .. note::
      This method is constrained to only use
      models with a ``.fit()``-method.

    Parameters
    ----------
      name : str
        name of the entry in the :class:`model_dict` and :class:`model_score`
      data : tuple
        tuple with training and test data in the order (X_train, y_train, X_test, y_test)
      models : list
        list of tuples with models and a dict with parameters

    Returns
    -------
      model_dict : dict
        dictionary containing all models trained
      model_scores : dict
        dictionary containing lists with scores for each model trained
    """
    
    X_train, y_train, X_test, y_test = data[0], data[1], data[2], data[3]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    if models is None:
      # choosing default set of models and parameters
      models = [(RandomForestClassifier, {'n_estimators': 100, 'n_jobs': -1})
                , (GradientBoostingClassifier, {})
                , (KNeighborsClassifier, {'n_jobs': -1})
                , (LogisticRegression, {'solver' :'saga', 'n_jobs': -1})]
    else:
      # checking that the input is correct
      for mod in models:
        assert isinstance(mod[1], dict), "You must provide the model's parameters in a dictionary."
    
    model_dict = {}
    model_scores = {name: []}
    
    for mod, params in models:
      print("Training model {} ...".format(mod.__name__))
      
      clf, s = self._trainSingleModel(mod, params, X_train, y_train, X_test, y_test)

      model_dict[name] = {mod.__name__ : clf}
      model_scores[name].append(s)

    return model_dict, model_scores

  def _trainSingleModel(self, model, params, X_train, y_train, X_test, y_test):
    """Private method for training a single model and returning
    the model and its score

    Parameters
    ----------
      model : object
        ML-model-object with the methods ``.fit()`` and ``.predict()``
      params : dict
        dictionary with hyperparameter to the model
      X_train : array
        numpy-array with training data
      y_train : array
        numpy-array with training targets
      X_test : array
        numpy-array with test data
      y_test : array
        numpy-array with test targets

    Returns
    -------
      model_trained : object
        trained model
      scores : float
        test score for the model
    """
    from sklearn.metrics import accuracy_score
    
    assert hasattr(model, 'fit'), "The model {} doesn't have a .fit()-method.".format(model.__name__)
    
    model = model(**params)
    clf = model.fit(X_train, np.ravel(y_train)) # Using ravel() since sklearn doesn't like arrays of shape (m, 1)
    pred = clf.predict(X_test)
    score = accuracy_score(pred, np.ravel(y_test))

    return model, score

  def trainSynthTestReal(self
                        , scores=None
                        , target=None
                        , features=None
                        , test_size=0.2
                        , eval_size=0.2
                        , tstr_name='tstr'):
    """Method for training model(s) on synthetic data and test on real data.

    Parameters
    ----------
      scores : array
        numpy-array with scores from TSTR
    """
    name1 = self.name1
    name2 = self.name2

    if scores is None:
      # Loading synthetic training data
      X_train, _, y_train, _, _, _ = self.dataPrep(target
                                                    , features
                                                    , test_size
                                                    , eval_size
                                                    , name1)
      # Loading real test data
      _, _, _, _, X_test, y_test = self.dataPrep(target
                                                    , features
                                                    , test_size
                                                    , eval_size
                                                    , name2)

      data = (X_train, y_train, X_test, y_test)

      _, scores = self._trainSeveralModels(tstr_name, data)

    tstr = np.mean(scores[tstr_name])

    return tstr
