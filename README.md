DACO - Dataframe Comparing tool
===============================

You are now trapped inside the repository containing the beginnings of a package/class for comparing two Pandas dataframes in Python. The goal is to be able to compare dataframes in several contexts, for example checking whether a synthetic dataset is good enough for your purposes.

The docs are found [here](https://navikt.github.io/ai-lab-daco/).

The idea is to use it like this:
```python
from daco import daco

# load your data frames
df1 = ...
df2 = ...

# create the daco-object
daco_obj = daco(df1, df2, name1='real_df', name2='fake_df')

# calculate distributions for all variables/columns in the dataframes
dist = daco_obj.findDistributions()

# do the chisquare-test on one variable
D, p = daco_obj.chisquare(var1)

# a dictionary will then be filled with values
print(daco_obj.p_D_chisquare)
```
The last line will produce the output on the form
```python
{var1: {'D': <number>, 'p': <number>}}
```