.. _usage:

Using DACO 
==========

The idea is to use it like this:

.. high-light: python

::

  from daco import daco

  # load your data frames
  df1 = ...
  df2 = ...
  var1 = ...

  # create the daco-object
  daco_obj = daco(df1, df2, name1='real_df', name2='fake_df')

  # calculate distributions for all variables/columns in the dataframes
  dist = daco_obj.findDistributions()

  # do the chisquare-test on one variable
  D, p = daco_obj.chisquare(var1)

  # a dictionary will then be filled with values
  print(daco_obj.p_D_chisquare)

The last line will produce the output on the form

::
  
  {var1: {'D': <number>, 'p': <number>}}

For detailed documenation of funtionality in DACO, see :ref:`daco_full_doc`.