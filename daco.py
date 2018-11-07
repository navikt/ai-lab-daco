import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    self.p_D_chisquare = {} # empty dict for saving chisquare values

  def findDistributions(self, bins=10, range_=None, density=None):
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

    distributions = {}

    # looping over all dataframes and their corresponding names
    for df in [ (df1, name1), (df2, name2) ]:
      hist = {}
      for column in df[0]:
        hist[ str(column) ] = np.histogram( df[0][ column ], bins=bins, range=range_, density=density )
      distributions[ df[1] ] = hist

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

    dist1 = distributions['df1'][var1][0]
    dist2 = distributions['df2'][var1][0]

    D, p = scipy.stats.chisquare( dist1, dist2 )

    # Saving results in dictionary
    p_D_chisquare[var1] = { 'D': D, 'p': p }

    return D, p


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
                                  , title=''
                                  , bins=50
                                  , range_hist=None
                                  , all_in_one_plot=False):
    """Funksjon som returner plott av distribusjoner til variable
    i dataene du gir til funksjonen.

    args:
        data: (list of arrays eller array) data som du vil plotte distribusjonen til
        filename: (str) navn på plottfil
        all_in_one_plot: (bool) hvis True blir histogram av alle arrays plottet i samme figur
        file_dir: (str) hvor plottfilen skal lagres
    """

    df1       = self.df1
    file_dir  = self.file_dir
    distributions = self.distributions

    if not all_in_one_plot:
      plt.figure()
      plt.hist(df1[variable], bins=bins, range=range_hist, density=True)
      plt.xlabel(xlabel)
      plt.title(title)
      plt.savefig(file_dir + filename)
      plt.close()
      return
    count = 1
    for array in df1:
      plt.figure(count)
      plt.hist(array, bins=50, density=True)
      plt.xlabel(xlabel)
      plt.title(title)
      plt.savefig(file_dir + str(count) + '_' + filename)
      count += 1
    plt.close()