#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module:: daco_privacy
   :platform: Unix, macOS
   :synopsis: Privacy module for DACO.

.. moduleauthor:: Jon Vegard Sparre
                  Robindra Prabhu

.. todo::
    - plotting/smart plotting, i.e. show only anomalies
    - pull plots based on the output from pd.dataframe.describe
    - allow setting range, density, binning, etc. for each variable manually?

    - add legends on boxplot for distance metrics
"""

import pandas as pd

class privacy:

  def __init__(self):
    pass

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
    
    disclosure_risks = {}
    disclosure_risks['true_match_max'] = count_matches.true_matches.max() # Plukker ut person/ekvivalensklasse med høyest andel av true matches. 
    disclosure_risks['true_match_mean'] = count_matches.true_matches.mean()
    disclosure_risks['true_match_median'] = count_matches.true_matches.median()
    disclosure_risks['true_match_data'] = count_matches.true_matches
    
    self.disclosure_risks = disclosure_risks