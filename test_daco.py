from daco import daco

import pandas as pd

header = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country', 'pred_var']
df = pd.read_csv('~/github/ai-lab-daco/datasets/adult.data.txt', sep=",",names=header, header=None)

cat_var = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country', 'pred_var']
for var in (set(header) - set(cat_var)):
  df[var] = df[var].astype('float')

for var in cat_var:
  df[var] = df[var].astype('category')

print(df.info())

df1 = df[:15000]
df2 = df[15000:]

daco_obj = daco(df1,df2)
dist = daco_obj.findDistributions()
daco_obj.logisticRegressionBenchmark(features=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
                                      , target=['pred_var'])
# daco_obj.plotCanvas()

# print(daco_obj.ks2_test('age'))
# print(daco_obj.ks2_test_val)