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


features_columns = ['triad_'+("%02d" % (number,)) for number in range(1,14)]


with col1:

  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    m0 = len(df)
    #remove rows with NaN values
    df = df.dropna(axis=0)
    m1 = len(df)
    st.subheader('◼ Raw data')
    st.write("**Note:** Rows with NaN entries have been removed.")
    st.write(df)
    st.subheader('◼ Descriptive Statistics')
    st.write(df.describe())

  else:
    st.sidebar.info('Upload a CSV file')



# Remove countries with zero triad counts?
if uploaded_file is not None:
  tf = st.sidebar.selectbox('Remove countries with zero entries:',
    [True,False],index = 1)
  if tf == True:
    for column in features_columns:
      df = df[df[column]!=0]
    m2 = len(df)

if uploaded_file is not None:
  st.sidebar.write(str(m0) +' initial countries')
  st.sidebar.write(str(m1)+' countries after removing NaN entries')
  if tf == True:
    st.sidebar.write(str(m2)+' countries after removing zero entries')


with col1:

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
      var1 = st.selectbox('feature 1',variables,index=14)
      # index 14 is triad_13
      var2 = st.selectbox('feature 2',variables,index=17)
      # index 17 is wgi.stability
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




st.markdown("--" * 34) 


if uploaded_file is not None:
  st.header('Linear Regression')


col1, col2 = st.columns([1,1])


# LINEAR REGRESSION
if uploaded_file is not None:

  # select measure to predict
  with col1:
    
    measure = st.selectbox(
      'Predicted measure:',
      ('wgi.corrupt','wgi.govt',
        'wgi.stability','wgi.regulatory',
        'wgi.law','wgi.voice'),
      index = 2 # index 2 = wgi.stability
      )

  # select features to use
#  features_columns = ['triad_'+("%02d" % (number,)) for number in range(1,14)]

  with col2:
    options = ['triad_'+("%02d" % (number,)) for number in range(1,14)]
    features_columns = st.multiselect(
      'Triads included:',
      options,
      options)

  if len(features_columns) > 0:

    y = np.array(df[measure])
    X = np.array(df[features_columns])

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

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
    if len(features_columns) > 0:
      st.write(fig_lin_reg)










