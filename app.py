import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

import math
import numpy as np

import altair as alt

st.title('National Stability Data')
st.write('enrico.borriello@asu.edu - Latest update: Mar 29, 2023')


#f = open('email.txt', 'r')
#content = f.read()
#with st.expander("Shade's email"):
#    st.write(content)

# IMPUT FILE
st.sidebar.subheader('Input CSV')
uploaded_file = st.sidebar.file_uploader("Choose a file")


col1, col2 = st.columns([1,1])


with col1:

  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    #remove rows with NaN values
    df = df.dropna(axis=0)
    st.subheader('◼ Raw data')
    st.write("**Note:** Rows with NaN entries have been removed.")
    st.write(df)
    st.subheader('◼ Descriptive Statistics')
    st.write(df.describe())

  else:
    st.sidebar.info('Upload a CSV file')


  # CORRELATION MATRIX
  if uploaded_file is not None:
    data = df.drop(['yr','harmonized.name'],axis=1)
    fig_corr_matrix, ax = plt.subplots()
    sb.heatmap(data.corr(), cmap="YlGnBu", annot=False)


  # SCATTER/HISTOGRAM
  if uploaded_file is not None:
    variables = np.array(df.columns)
    with col2:
      st.subheader('◼ Correlation/Histogram')
      var1 = st.selectbox('feature 1',variables)
      var2 = st.selectbox('feature 2',variables)
    fig_sc_hist, ax = plt.subplots()
    if var1 == var2:
        sb.histplot(df, x = var1)
    else:
        sb.scatterplot(df, x = var1, y = var2)
    with col2:
      st.write(fig_sc_hist)

with col2:
  if uploaded_file is not None:
    st.subheader('■ Correlation Matrix')
    st.write(fig_corr_matrix)





st.header('Regressions')


col1, col2 = st.columns([1,1])



# LINEAR REGRESSION
if uploaded_file is not None:

  # select measure to predict
  with col1:
    st.subheader('◼ Linear Regression')
    measure = st.selectbox(
      'Select the measure to predict',
      ('wgi.corrupt','wgi.govt',
        'wgi.stability','wgi.regulatory',
        'wgi.law','wgi.voice')
      )

  # select features to use
  features_columns = ['triad_'+("%02d" % (number,)) for number in range(1,14)]
  y = np.array(df[measure])

  # design matrix
  X = np.array(df[features_columns])
  ones = [[1]]*len(X)
  Xd = np.hstack((ones,X));

  # find parameters
  theta = np.dot(np.dot(np.linalg.inv(np.dot(Xd.T,Xd)),Xd.T),y)

  # preditions
  predictions = np.dot(Xd,theta)

  # plot
  fig_lin_reg, ax = plt.subplots()
  ax.grid(b=True, which='major', color='lightgray', linestyle='-')
  ax.set_xlim((-3,3))
  ax.set_ylim((-3,3))
  ax.set_aspect(1)

  plt.scatter(y,predictions,edgecolors='white',s=60)
  plt.plot([-3,3],[-3,3],'--')
  plt.xlabel('actual',size=12)
  plt.ylabel('predicted',size=12)

  # RMSE
  MSE = np.square(np.subtract(y,predictions)).mean() 
  RMSE = math.sqrt(MSE)
  plt.title(str(measure)+'      '+'RMSE = '+str(round(RMSE,3)))

with col1:
  if uploaded_file is not None:
    st.write(fig_lin_reg)


with col2:
  if uploaded_file is not None:
    st.empty()







