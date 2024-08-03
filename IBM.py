import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st


"""Importing Dataset"""

medical_df = pd.read_csv('insurance.csv') #import dataset

warnings.filterwarnings("ignore")

medical_df.head()

"""We can see charges is the target variable.

Analysing Data"""

medical_df.shape #gives us rows and columns

medical_df.info() #gives us dataTypes

medical_df.describe() #gives numerical data(statistical measures)

medical_df.isnull().sum() #gives us total null values in every column

"""Data Visualizaion"""

plt.figure(figsize=(3,3))
sns.displot(data = medical_df, x = 'age')

plt.figure(figsize=(3,3))
sns.displot(data = medical_df, x = 'sex',kind = 'hist')

medical_df['sex'].value_counts()

medical_df.info()

plt.figure(figsize=(4,4))
sns.displot(data=medical_df,x='bmi')
plt.show()

medical_df['bmi'].value_counts()

plt.figure(figsize=(4,4))
sns.countplot(medical_df['children'])
plt.show()

medical_df['children'].value_counts()

plt.figure(figsize=(4,4))
sns.countplot(data=medical_df,x='smoker')
plt.show()

medical_df.head()

"""Convert categorical column to numerical"""

medical_df['region'].value_counts()

medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
medical_df.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)

medical_df.head()

"""Train Test Split"""

X = medical_df.drop('charges',axis=1)
y = medical_df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

#X_train.shape

#X_test.shape

lg = LinearRegression()
lg.fit(X_train,y_train) # 80 model will be train
y_pred = lg.predict(X_test) # 10 model will be predicted

r2_score(y_test,y_pred)

"""Prediction"""


st.title("Medical Insurance Prediction Model")
input_text = st.text_input("Enter Person All Feature")
input_text_splitted = input_text.split(",")
try:
    np_df = np.asarray(input_text_splitted,dtype=float)
    prediction = lg.predict(np_df.reshape(1,-1))
    st.write("Medical Insurance is for this person is :\n",prediction[0])
except ValueError:
    st.write("Please enter numerical value")
