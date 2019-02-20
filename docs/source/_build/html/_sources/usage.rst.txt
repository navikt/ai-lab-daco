.. _usage:

Using DACO 
==========

The idea is to use it like this:

.. high-light: python

::

  from daco.daco import daco

  # load your data frames
  df1 = ...
  df2 = ...
  # Choose one of the variables in the dataset to compute statistics of
  var1 = 'foo'

  # create the daco-object
  daco_obj = daco(df1, df2, name1='real_df', name2='synth_df')

  # calculate distributions for all variables/columns in the dataframes
  dist = daco_obj.findDistributions()

  # do the chisquare-test on one variable
  D, p = daco_obj.chisquare(var1)

  # a dictionary will then be filled with values
  print(daco_obj.p_D_chisquare)

  # plot a canvas with distributions of all the variables
  daco_obj.plotCanvas()

The results from ``chisquare`` will be outputted on the form

::
  
  {var1: {'D': <number>, 'p': <number>}}

For detailed documenation of funtionality in DACO, see :ref:`daco_main`.