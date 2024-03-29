**DACO - Dataframe Comparing tool**
===============================
 
[![Known Vulnerabilities](https://snyk.io/test/github/navikt/ai-lab-daco/badge.svg)](https://snyk.io/test/github/navikt/ai-lab-daco) [![GitHub version](https://badge.fury.io/gh/navikt%2Fai-lab-daco.svg)](https://badge.fury.io/gh/navikt%2Fai-lab-daco)

You are now trapped inside the repository containing the beginnings of a package/class for comparing two Pandas dataframes in Python. The goal is to be able to compare dataframes in several contexts, for example checking whether a synthetic dataset is good enough for your purposes.

The full docs are found [here](https://navikt.github.io/ai-lab-daco/).

## Install
1. Clone this repo
2. ``cd`` into this repo
3. Run ``pip install .``
4. Voilà!


## Short how-to

The idea is to use it like this:
```python
import daco

# load your data frames
df1 = ...
df2 = ...

# create the daco-object
daco_obj = daco.daco(df1, df2, name1='real_df', name2='fake_df')

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
