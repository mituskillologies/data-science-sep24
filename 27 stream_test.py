import streamlit as st
import pandas as pd
import numpy as np

st.title('Salary Prediction')

st.write('this app predicts the salary based on years of experience')



df = pd.read_csv('Salary_Data.csv')

x = df['YearsExperience'].values.reshape(30,1)

y = df['Salary']

df.columns

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)


st.sidebar.title('INPUT')
exp = st.sidebar.slider("years of exp", min_value= float(df['YearsExperience'].min()),
                        max_value= float(df['YearsExperience'].max()),
                        value= float(df['YearsExperience'].min()))


input_data = [[exp]]


# prediction

pred = lr.predict(input_data)

st.write(f"salary is {pred}")