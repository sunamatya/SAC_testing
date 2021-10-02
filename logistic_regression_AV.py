import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('AV_outdata_30.csv')
#df = pd.read_csv('AV_outdata.csv')
#df = pd.read_csv('AV_outdata_200_var_cost_half.csv')
#df = pd.read_csv('AV_outdata_erratic.csv')

#df = df.dropna(subset=['agg'])
print(df.head())
print(df.describe())
print(df.info())
# #plt.figure(figsize =(14, 6))
#df = df.drop(df.columns[[15, 17]], axis=1)

print(df.head())
print(df.describe())
print(df.info())
#df['infc1'].replace(["TRUE","FALSE"], [1,0], inplace=True)
df['inf'].replace(["TRUE","FALSE"], [1,0], inplace=True)
# df['infc2'].replace(["TRUE","FALSE"], [1,0], inplace=True)
#sns.pairplot(df, hue='infc1')
sns.pairplot(df, hue='inf')
plt.show()
# final_df = df[df['Species'] != 'Iris-virginica']
# final_df = final_df.drop(final_df[(final_df['Species'] == "Iris-setosa") & (final_df['SepalWidthCm'] < 2.5)].index)
# sns.pairplot(final_df, hue='Species')
# #plt.show()
#
# df['infc1'].replace(["TRUE","FALSE"], [1,0], inplace=True)
# df['infc2'].replace(["TRUE","FALSE"], [1,0], inplace=True)


inp_df = df.drop(df.columns[[0,15]], axis=1)
out_df = df.drop(df.columns[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]], axis=1)
print(inp_df.info())
# #
scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)
#
X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)
X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.to_numpy()
y_ts_arr = y_test.to_numpy()

print('Input Shape', (X_tr_arr.shape))
print('Output Shape', X_test.shape)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_tr_arr, y_tr_arr)
print (clf.intercept_, clf.coef_)
pred = clf.predict(X_ts_arr)
print ('Accuracy from sk-learn: {0}'.format(clf.score(X_ts_arr, y_ts_arr)))
#
#
import statsmodels.api as sm

# inp_df_tester = df.drop(df.columns[[0,1,2,3,4,5,6,16]], axis=1)
# scaler = StandardScaler()
# inp_df_tester = scaler.fit_transform(inp_df_tester)
#
# logistic_regression = sm.Logit(out_df, inp_df_tester)
# #logistic_regression = sm.Logit(out_df, inp_df)
# result = logistic_regression.fit_regularized()
# print(result.summary())
#
# inp_df_tester = df.drop(df.columns[[0,7,8,9,10,11,12,13,14,15,16]], axis=1)
# scaler = StandardScaler()
# inp_df_tester = scaler.fit_transform(inp_df_tester)
#
# logistic_regression = sm.Logit(out_df, inp_df_tester)
# #logistic_regression = sm.Logit(out_df, inp_df)
# result = logistic_regression.fit_regularized()
# print(result.summary())


logistic_regression = sm.Logit(y_train, X_tr_arr)
#logistic_regression = sm.Logit(out_df, inp_df)
#result = logistic_regression.fit_regularized()
result = logistic_regression.fit_regularized()
yhat = result.predict(X_ts_arr)
pred = list(map(round, yhat))
#
from sklearn.metrics import (confusion_matrix, accuracy_score)
print('Test_accuracy =', accuracy_score(y_test, pred))
print(result.summary())




#
#
# import statsmodels.api as sm
#
#




