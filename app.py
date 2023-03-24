import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

st.title('National Stability Data')
st.write('enrico.borriello@asu.edu - Latest update: Mar 24, 2023')


f = open('email.txt', 'r')
content = f.read()


with st.expander("Shade's email"):
    st.write(content)


st.sidebar.subheader('Input CSV')
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  #remove rows with NaN values
  df = df.dropna(axis=0)
  st.subheader('Data')
  st.write("(I've removed rows with NaN entries.)")
  st.write(df)
  st.subheader('Descriptive Statistics')
  st.write(df.describe())

else:
  st.sidebar.info('Upload a CSV file')

if uploaded_file is not None:
  data = df.drop(['yr','harmonized.name'],axis=1)
  fig, ax = plt.subplots()
  sb.heatmap(data.corr(), cmap="YlGnBu", annot=False)
  st.subheader('Correlation Matrix')
  st.write(fig)

if uploaded_file is not None:
   variables = np.array(df.columns)
   var1 = st.sidebar.selectbox('feature 1',variables)
   var2 = st.sidebar.selectbox('feature 2',variables)
   fig, ax = plt.subplots()
   if var1 == var2:
       sb.histplot(df, x = var1)
   else:
       sb.scatterplot(df, x = var1, y = var2)
   st.subheader('Correlation/Histogram')
   st.write(fig)







