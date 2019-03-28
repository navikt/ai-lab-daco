# pylint: disable=C0103
"""Test script for daco.
"""

import pandas as pd

from src.daco_main import daco

header = ["sex","age","socprof","income","marital","depress","sport","nofriend","smoke","nociga","alcabuse","bmi"]   
df1 = pd.read_csv('~/github/ai-lab-daco/datasets/original_SD2011.csv', sep=",")#,names=header, header=None)
df1 = df1[df1.columns[1:]] 
df2 = pd.read_csv('~/github/ai-lab-daco/datasets/synthetic_SD2011.csv', sep=",")#,names=header, header=None))

cat_var = ["sex","socprof","marital","depress","sport","nofriend","smoke","nociga","alcabuse"]  
for var in set(header) - set(cat_var):
  df1[var] = df1[var].astype('float')
  df2[var] = df2[var].astype('float')

for var in cat_var:
  df1[var] = df1[var].astype('category')
  df2[var] = df2[var].astype('category')

features = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
target = ['pred_var']


# daco_obj.numericalComparing()
daco_obj = daco(df1,df2)
dist = daco_obj.findDistributions()
# daco_obj.plotDistanceMetrics()
daco_obj.plotDistributionsOfVariable('age')
# daco_obj.plotPairplot()
# hrr = daco_obj.hellingerRowForRow()

# daco_obj.logisticRegressionBenchmark(features=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
#                                       , target=['pred_var'])
# daco_obj.plotCanvas()
exit()
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


# plt.figure(figsize=(10, 6))
# ax = plt.subplot(1, 1, 1)
# df1.hist('fnlwgt', ax=ax, alpha=0.5, normed=True)
# df2.hist('fnlwgt', ax=ax, alpha=0.5, color="#e74c3c", normed=True)
# plt.show()