# -*- coding: utf-8 -*-
"""Water Feasibility Classification XGBoost
"""

import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

train_df = pd.read_csv('../path')
test_df = pd.read_csv('../path')

train_df.info()

missing = train_df.isna().mean(axis=0)
missing

train_df.isnull().sum()

train_df.duplicated().sum()

X_train = train_df.drop(["id","DC201"], axis = 1)
X_train_con = X_train[['DC216',"DC220","DC142a"]]
X_train_cat = X_train.drop(['DC216',"DC220","DC142a"], axis=1)

y_train = train_df.DC201
X_test = test_df.drop("id", axis = 1)
X_test_con = X_test[['DC216',"DC220","DC142a"]]
X_test_cat = X_test.drop(['DC216',"DC220","DC142a"], axis=1)

X_train_con = X_train_con.fillna(train_df.median())
X_train_cat = X_train_cat.fillna(train_df.median())
y_train = y_train.fillna("Layak Minum")

X_test_con = X_test_con.fillna(train_df.median())
X_test_cat = X_test_cat.fillna(train_df.median())

X_train_con.isnull().sum()

X_train_cat.isnull().sum()

y_train.isnull().sum()

train_cat = pd.concat([X_train_cat,y_train], axis=1)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

train_cat = train_cat.apply(LabelEncoder().fit_transform)
X_test_cat = X_test_cat.apply(LabelEncoder().fit_transform)

train_df = pd.concat([X_train_con, train_cat], axis=1)
test_df = pd.concat([X_test_con, X_test_cat], axis=1)

colnames = list(train_df.columns.values.tolist())
for col in colnames:
  print(col)
  print(train_df[col].unique())

colnames = list(test_df.columns.values.tolist())
for col in colnames:
  print(col)
  print(test_df[col].unique())

X_train = train_df.drop(["DC201"], axis = 1)
y_train = train_df.DC201
X_test = test_df

import xgboost as xgb

model = xgb.XGBClassifier(scale_pos_weight=9)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(preds)

sum(preds==1)

sum(preds==0)

from sklearn.metrics import f1_score

pred = model.predict(X_train)
f1 = f1_score(y_train, pred>0.5, average='weighted')
print ("SCORE:", f1)

from sklearn.metrics import accuracy_score

pred = model.predict(X_train)
accuracy = accuracy_score(y_train, pred>0.5)
print ("SCORE:", accuracy)

from sklearn import metrics
k = pd.DataFrame(metrics.confusion_matrix(y_train,pred))
print(k)

from sklearn.metrics import f1_score

pred = model.predict(X_train)
f1 = f1_score(y_train, pred>0.5, average='weighted')
print ("SCORE:", f1)

sum(preds==1)

sum(preds==0)

submission = pd.read_csv('../path')
submission

pred = pd.DataFrame(preds)
pred

ID = pd.DataFrame(submission["id"])
ID

result = pd.concat([ID, pred], axis=1, join='inner')
result

result.columns.values[1] = "DC201"
result

result = result.replace({0:"Layak Minum",1:"Tidak Layak Minum"})
result

result = result.reset_index(drop=True)
result

pd.DataFrame(result).to_csv('/content/drive/MyDrive/DSC/prediction_xgboost.csv')

from imblearn.over_sampling import SMOTE
import numpy as np

np.random.seed(2023)
X_oversampled, y_oversampled = SMOTE().fit_resample(X_train, y_train)

model.fit(X_oversampled, y_oversampled)
preds = model.predict(X_test)

print(preds)

sum(preds==1)

sum(preds==0)

pred = model.predict(X_oversampled)
f1 = f1_score(y_oversampled, pred>0.5, average='weighted')
print ("SCORE:", f1)

pred = model.predict(X_oversampled)
accuracy = accuracy_score(y_oversampled, pred>0.5)
print ("SCORE:", accuracy)

k = pd.DataFrame(metrics.confusion_matrix(y_oversampled,pred))
print(k)
