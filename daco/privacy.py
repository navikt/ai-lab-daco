#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module:: privacy
   :platform: Unix, macOS
   :synopsis: Privacy module for DACO.

.. moduleauthor:: Jon Vegard Sparre
                  Robindra Prabhu

.. todo::
    - this class should be independent of other daco-functionality, how?
"""

import pandas as pd
import numpy as np

class privacy:
  """Class with metrics for measuring privacy risks.
  """
  
  def __init__(self):
    pass

  def attributeDisclosure(self, sensitive_var, other_vars=[], df1=None, df2=None):
    """Method doing an attribute disclosure-calculation.
    
    Idea: User gives a list with sensitive variables which an intruder may want to learn, this
    method will use the remaining variables to create subsets of the synthetic dataset
    w.r.t. each person in original dataset and look at how equal the sensitive variables
    are.
    """

    if (df1 is None) and (df2 is None):
      df1 = self.df1
      df2 = self.df2
    
    if len(other_vars) == 0:
      other_vars = set(sensitive_var).symmetric_difference(set(df1.columns))
    
    count_matches = pd.DataFrame(columns=['index', 'matches', 'n_values', 'true_matches'])
    
    # Assuming each row is a unique person
    for index, person in df1.iterrows():
      person_values = person[other_vars].values 
      subset_idx = (df2[other_vars] == person_values).all(axis=1)
      subset = df2[subset_idx] # subset = ekvivalensklasse
      n_values = subset[sensitive_var].nunique() # l-diversity
      true_matches = (subset[sensitive_var] == person[sensitive_var]).sum() / subset.shape[0] # Andel i subset som også matcher på sensitiv variabel.
      count_matches = count_matches.append([{'index': index
                                            , 'matches': subset.shape[0]
                                            , 'n_values': n_values
                                            , 'true_matches': true_matches}])
    
    self.matches = count_matches
    
    disclosure_risks = {}
    disclosure_risks['true_match_max'] = count_matches.true_matches.max() # Plukker ut person/ekvivalensklasse med høyest andel av true matches. 
    disclosure_risks['true_match_mean'] = count_matches.true_matches.mean()
    disclosure_risks['true_match_median'] = count_matches.true_matches.median()
    disclosure_risks['true_match_data'] = count_matches.true_matches
    
    self.disclosure_risks = disclosure_risks

def correctRelativeAttributionProbability(self, sensitive_var, key_variables=[], df1=None, df2=None):
  """Method for calculating attribute disclosure as defined by Gillian M Raab
  https://www.geos.ed.ac.uk/homes/graab/simons_march1832019.pdf

  - Define a set of key variables that an intruder might know -> var
  - Select an attribute that might be sensitive or check out all
  - Tabulate the attribute by all combinations of the key in the synthetic data
  - Calculate the % of all rows in the synthetic data that are attribute disclosive and are also represented in the original data P_disclosive
  - From the original check whether the attribution is correct and get P_correct: CRAP = P_disclosive *p_correct
  """

  if (df1 is None) and (df2 is None):
      df1 = self.df1
      df2 = self.df2
    
  if len(other_vars) == 0:
    other_vars = set(sensitive_var).symmetric_difference(set(df1.columns))
  
  count_matches = pd.DataFrame(columns=['index', 'matches', 'n_values', 'true_matches'])

  # Different measures are used on categorical and numerical features
  other_cat = set(other_vars).issubset(self.cat_var)
  other_num = set(other_vars).issubset(self.num_var)
  
  # Assuming each row is a unique person
  for index, person in df1.iterrows():
    person_values_num = person[other_num].values
    person_values_cat = person[other_cat].values
    # numerical values must be equal within an epsilon, categorical values must be measured by amount of equal characters?
    # subset_idx = (df2[other_vars] == person_values).all(axis=1)
    subset_idx_num =  np.isclose(df2[other_num], person_values_num, atol=0.1) # atol sets the tolerance for how close the values must be in order to evaluate to True
    subset_idx_cat = (df2[other_cat] == person_values_cat).all(axis=1)
    subset_idx = subset_idx_cat + subset_idx_num
    subset = df2[subset_idx] # subset = ekvivalensklasse
    n_values = subset[sensitive_var].nunique() # l-diversity
    true_matches = (subset[sensitive_var] == person[sensitive_var]).sum() / subset.shape[0] # Andel i subset som også matcher på sensitiv variabel.
    count_matches = count_matches.append([{'index': index
                                          , 'matches': subset.shape[0]
                                          , 'n_values': n_values
                                          , 'true_matches': true_matches}])
  
  self.matches = count_matches
  
  disclosure_risks = {}
  disclosure_risks['true_match_max'] = count_matches.true_matches.max() # Plukker ut person/ekvivalensklasse med høyest andel av true matches. 
  disclosure_risks['true_match_mean'] = count_matches.true_matches.mean()
  disclosure_risks['true_match_median'] = count_matches.true_matches.median()
  disclosure_risks['true_match_data'] = count_matches.true_matches
  
  self.disclosure_risks = disclosure_risks