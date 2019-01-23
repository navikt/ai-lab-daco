# pylint: disable=C0103
"""Test script for daco.
"""

import pandas as pd

from daco.daco import daco

header = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country', 'pred_var']
df = pd.read_csv('~/github/ai-lab-daco/datasets/adult.data.txt', sep=",",names=header, header=None)

cat_var = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country', 'pred_var']
for var in set(header) - set(cat_var):
  df[var] = df[var].astype('float')

for var in cat_var:
  df[var] = df[var].astype('category')

features = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
target = ['pred_var']

df1 = df[:15000]
df2 = df[15000:]
# for var in (set(header) - set(cat_var)):
#   df1[var] = df1[var]*np.random.randint(1,19999)  

# daco_obj.numericalComparing()
daco_obj = daco(df1,df2)
daco_obj.plotPairplot()
exit()
# dist = daco_obj.findDistributions()
hrr = daco_obj.hellingerRowForRow()

# daco_obj.logisticRegressionBenchmark(features=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
#                                       , target=['pred_var'])
# daco_obj.plotCanvas()
# daco_obj.plotDistributionsOfVariable('age')
# daco_obj.plotPairplot()
# daco_obj.rowMatching()
# print(daco_obj.match_values)
# daco_obj.plotCorrelation()
# daco_obj.logisticRegressionBenchmark(features=features, target=target)
# from sklearn.neighbors import KNeighborsClassifier
# models, scores = daco_obj.trainModels(features=features, target=target, models=[(KNeighborsClassifier, {})])
# model_dict, model_scores = daco_obj.trainAndTestModels(features=features, target=target)
# sra = daco_obj.syntheticRankingAgreement(model_scores=model_scores)
# tstr = daco_obj.trainSynthTestReal(features=features, target=target)
# print(tstr)

# print(daco_obj.ks2_test('age'))
# print(daco_obj.ks2_test_val)