from daco import daco

import pandas as pd


header = [
          "A1 - Checking account" # status of exisiting checking account
         ,"A2 - Duration(month)" # duration in month
         ,"A3 - Credit history" #credit history
         ,"A4 - Purpose" # what is the loan for?
         ,"A5 - Credit amount" # how much is asked for?
         ,"A6 - Savings account" #status of savings account/bonds
         ,"A7 - Employment duration" # present employment since
         ,"A8 - Installment rate" # installment rate in percentage of disposable income
         ,"A9 - Civil status and gender" # personal status and gender
         ,"A10 - Debtors/guarantors" # other debtors / guarantors?
         ,"A11 - Present residence since" # present residence since?
         ,"A12 - Property" # property
         ,"A13 - Age" # age in years
         ,"A14 - Other installment plans" # other installment plans?
         ,"A15 - Housing" # Housing: rent, own, for free?
         ,"A16 - Existing credits" # number of existing credits at this bank
         ,"A17 - Job" # unemployed? unskilled? self-employed? highly qualified employee? etc.
         ,"A18 - Liability" # number of people liable to provie maintenance for
         ,"A19 - Telephone" # has a phone?
         ,"A20 - Foreign worker" # is a foreign worker?
         ,"Score" # 1 = Good, 2 = Bad
        ]

df1 = pd.read_csv('~/github/ai-lab-daco/datasets/xaa-numeric.csv', sep="\s+",names=header, header=None)
df2 = pd.read_csv('~/github/ai-lab-daco/datasets/xab-numeric.csv', sep="\s+",names=header, header=None)

daco_obj = daco(df1[header[-20:]], df2[header[-20:]])
dist = daco_obj.findDistributions()

D, p = daco_obj.chisquare('A6 - Savings account')

# dict with D and p values now available as attribute:
daco_obj.p_D_chisquare
