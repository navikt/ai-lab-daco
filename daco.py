"""
NOTES:

Kolmogorov-Smirnov
t-test
chi2
B-dist

Fint sted å hente inspirasjon til dokumentering:
https://realpython.com/documenting-python-code/
http://www.sphinx-doc.org/en/stable/index.html

.. moduleauthor Jon Vegard Sparre
                Robindra Prabhu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scipy
class daco():
  """ Class for comparing two Pandas dataframes statistically.

  The purpose of this class is to easily compare datasets in different
  settings, e.g. check if a synthetic version of a dataset is good enoug
  for your use.

  ...

  Attributes
  ----------

  TODO
    - sequential vs tabular data (long term)
    - differences (short term)
    - distances (short term)
    - correlations (short term) (global metric)
      - diff between correlations
    - local and global metrics
    - binning
    - differential privacy (long term)
    - privacy checks
    - mean, variance, ...
    - logistic regression (accuracy, confusion matrix, feature importance)
    - plotting
      - smart plotting, i.e. show only anomalies
    - checks for dataframe (e.g. it must have a header, the column names must be equal)
    - pull plots based on the output from pd.dataframe.describe
  """
  def __init__( self, df1, df2, name1='df1', name2='df2', file_dir="plots/"):
    """
    Args:
      df1, df2 (dataframe): dataframes to be compared. Must have header and
        dtype for all columns.
      file_dir (str): path to directory where plots are saved
      name1, name2 (str): names of dataframes, used as keys in dictionaries
        containing info about dataframes
    """
    self.df1      = df1
    self.df2      = df2
    self.file_dir = file_dir
    self.name1 = name1
    self.name2 = name2
    # Creating dicts for saving values for different metrics
    self.p_D_chisquare = {}
    self.bhattacharyya_dis = {}
    self.hellinger_div = {}
    self.kullbackleibler_div = {}

  def findDistributions(self, bins_='auto', range_=None, density=True):
    """Find distributions of all variables/columns in dataframes loaded
    into daco.

    Args:
      bins (int): number of bins in histogram/distribution
      range_ (tuple): max and min limit for range of distribution
      density (bool): if True a normalised distribution is returned

    Returns:
      distributions (dict): dictionary with dictionaries containing
        all distributions in numpy arrays.
    """
    df1 = self.df1
    df2 = self.df2
    name1 = self.name1
    name2 = self.name2

    hist1 = {}
    hist2 = {}

    # looping over all columns containing numerical variables
    column_numerical = df1.select_dtypes(include=[np.number]).columns
    for column in column_numerical:
      min_val = min(df1[ column ].min(), df2[ column ].min())
      max_val = max(df1[ column ].max(), df2[ column ].max())
      range_ = (min_val, max_val)
      
      hist1[ str(column) ] = np.histogram( df1[ column ], bins=bins_, range=range_, density=density )
      hist2[ str(column) ] = np.histogram( df2[ column ], bins=hist1[ str(column) ][1], range=range_, density=density )

    # looping over all columns containing categorical variables
    column_categories = df1.select_dtypes(include=['category']).columns
    for column in column_categories:
      value_count1 = df1[column].value_counts(sort=False)
      value_count2 = df2[column].value_counts(sort=False)
      hist1[ str(column) ] = [ value_count1.values, value_count1.index.categories ]
      hist2[ str(column) ] = [ value_count2.values, value_count2.index.categories ]
    
    distributions = {}
    distributions[ name1 ] = hist1
    distributions[ name2 ] = hist2

    # the results should be available across the class
    self.distributions = distributions

    return distributions

  def chisquare(self, var1):
    """ Method for calculating the chisquare test.

    Args:
      var1 (str): name of variable to do the chisquare test. Must be contained
        in the dataframes.
    """
    distributions = self.distributions
    p_D_chisquare = self.p_D_chisquare

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    D, p = scipy.stats.chisquare( dist1, dist2 )

    # Saving results in dictionary
    p_D_chisquare[var1] = { 'D': D, 'p': p }

    return D, p

  def hellinger(self, var1):
    """ Calculate the Hellinger divergence for the distributions of
    var1 in the two dataframes.
    See https://en.wikipedia.org/wiki/Hellinger_distance

    Args:
      var1 (str): name of variable to do calculate the Hellinger divergence
        for. Must be contained in the dataframes.
    
    Returns:
      hellinger_div (float): the value of the Hellinger divergence.
    """
    distributions = self.distributions
    hellinger_div = self.hellinger_div

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    hellinger_div_value = np.sqrt( np.sum( ( np.sqrt(dist1) - np.sqrt(dist2) )**2 ) )

    hellinger_div[var1] = { 'hellinger': hellinger_div_value }

    return hellinger_div_value

  def kullbackleibler(self, var1):
    """ Calculate Kullback-Leibler divergence for the distributions of
    var1 in the two dataframes.
    See https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence
        https://en.wikipedia.org/wiki/Kullback–Leibler_divergence

    Args:
      var1 (str): name of variable to do calculate the Kullback-Leibler
        divergence for. Must be contained in the dataframes.
    
    Returns:
      kb_div (float): the Kullback-Leibler divergence
    """
    import scipy.special as spec

    distributions = self.distributions
    kullbackleibler_div = self.kullbackleibler_div

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    # return np.sum( np.where( p != 0., p * np.log( p / np.where( q != 0., q, 1 ) ), 0 ) )
    # return sp.entropy(p, q)

    kb_div = spec.kl_div(dist1, dist2)

    kullbackleibler_div[var1] = { 'kullback': kb_div }

    return kb_div

  def bhattacharyya(self, var1):
    """ Calculate the Bhattacharyya distance for the distributions of
    var1 in the two dataframes.
    See https://en.wikipedia.org/wiki/Bhattacharyya_distance

    Args:
      var1 (str): name of variable to do calculate the Bhattacharyya distance
        for. Must be contained in the dataframes.
    
    Returns:
      b_dis (float): the Bhattacharyya distance.
    """

    distributions = self.distributions
    bhattacharyya_dis = self.bhattacharyya_dis

    dist1 = distributions[self.name1][var1][0]
    dist2 = distributions[self.name2][var1][0]

    def normalize(h):
      return h/np.sum(h)

    b_dis = 1 - np.sum( np.sqrt( np.multiply( normalize(dist1), normalize(dist2) ) ) )

    bhattacharyya_dis[var1] = { 'bhattacharyya': b_dis }

    return b_dis

  def ks2_test(self, var):
    """Method using the scipy.stats.ks_2samp for computing the Kolmogorov-
    Smirnov statistic on two samples. The result is added to a dictionary
    keeping the results of all calculations.

    Args:
      var (str): name of variable in both dataframes you want to do
        the test for.
    """

  def plotCorrelation(self, xlabel="", ylabel="", title="", filename="correlations"):
    """Plotting correlations between columns in a dataframe
    
    args:
        X:         (dataframe) data du vil finne korrelasjonene mellom
        xlabel:    (str) label på x-aksen
        ylabel:    (str) label på y-aksen
        title:     (str) plottittel
        file_dir:  (str) hvor plottfilen skal lagres
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
                    , cbar_kws={ "fraction": .05, "shrink": .5, "orientation": "vertical" } )
    ax0.set_title(title)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    plt.savefig(file_dir + filename + ".png")
    plt.close()

  def plotDistributionsOfVariable(self
                                  , variable
                                  , filename=''
                                  , xlabel=''
                                  , title=''):
    """Funksjon som returner plott av distribusjoner til variable
    i dataene du gir til funksjonen.

    args:
        data: (list of arrays eller array) data som du vil plotte distribusjonen til
        filename: (str) navn på plottfil
        all_in_one_plot: (bool) hvis True blir histogram av alle arrays plottet i samme figur
        file_dir: (str) hvor plottfilen skal lagres
    """

    df1       = self.df1
    df2       = self.df2
    file_dir  = self.file_dir
    distributions = self.distributions

    if variable in df1.select_dtypes(include=[np.number]).columns:
      width_ = abs( max( df1[variable].max(), df2[variable].max() ) - min( df1[variable].min(), df2[variable].min() ) ) / len(distributions['df1'][variable][1])
      df1_err = np.sqrt(distributions['df1'][variable][0])
      df2_err = np.sqrt(distributions['df2'][variable][0])

      # ratio = distributions['df2'][variable][0] / distributions['df1'][variable][0]
      # ratio_err = np.sqrt( (df1_err/distributions['df1'][variable][0])**2 + (df2_err/distributions['df2'][variable][0])**2 )*ratio

      gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2,1])
      fig = plt.figure(0)
      ax1 = fig.add_subplot(gs[0])
      ax1.bar(distributions['df1'][variable][1][:-1], distributions['df1'][variable][0], align='edge', width=width_)#, yerr=df1_err)
      ax1.bar(distributions['df2'][variable][1][:-1], distributions['df2'][variable][0], align='edge', width=width_, alpha=0.5)
      ax1.set_title(variable)
      
      ax2 = fig.add_subplot(gs[1], sharex=ax1)
      ax2.axhline(y=1, color='k', linestyle='-')
      # ax2.errorbar(distributions['df2'][variable][1][:-1] + width_/2, ratio, yerr=ratio_err, fmt='o')
      ax2.set_ylim(0,2)
      plt.setp(ax1.get_xticklabels(), visible=False)
      plt.subplots_adjust(hspace=0)
      plt.xlabel(xlabel)
      plt.show()
      plt.close()

    else:
      ratio = distributions['df2'][variable][0] / distributions['df1'][variable][0]

      gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2,1])
      fig = plt.figure(0)
      ax1 = fig.add_subplot(gs[0])
      ax1.bar(distributions['df1'][variable][1], distributions['df1'][variable][0], align='center', width=1)#, yerr=df1_err)
      ax1.bar(distributions['df2'][variable][1], distributions['df2'][variable][0], align='center', width=1, alpha=0.5)
      ax1.set_title(variable)
      
      ax2 = fig.add_subplot(gs[1], sharex=ax1)
      ax2.axhline(y=1, color='k', linestyle='-')
      ax2.plot(distributions['df2'][variable][1], ratio, 'o')
      ax2.set_ylim(0,2)
      plt.xticks(rotation=90)
      plt.setp(ax1.get_xticklabels(), visible=False)
      plt.subplots_adjust(hspace=0)
      plt.xlabel(xlabel)
      plt.show()
      plt.close()

    return 0