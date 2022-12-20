# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd
#from sklearn.datasets import *
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

st.subheader("""
         ***Iris Flower Prediction***
         """)
         
st.sidebar.header('User Input Parameter')

def user_input_features():
    sepal_l = st.sidebar.slider('Speal Length', 2.01, 7.8, 5.5)
    sepal_w = st.sidebar.slider('Sepal Width', 2.02, 8.0, 4.2)
    petal_l = st.sidebar.slider('Petal Length', 1.01, 7.5, 4.5)
    petal_w = st.sidebar.slider('Petal Width', 0.01, 3.9, 2.0)
    
    features = {'Speal Length' : sepal_l,
                'Sepal Width' : sepal_w,
                'Petal Length' : petal_l,
                'Petal Width' : petal_w}
    data = pd.DataFrame(features, index=[0])
    return data

func = user_input_features()

st.write(""" **User Input Parameter** """)
st.write(func)

dataset = datasets.load_iris()
X = dataset['data']
Y = dataset['target']

# Creating Model
model = LogisticRegression()
model.fit(X, Y)

st.write(""" **Label with its Index Number** """)
st.write(dataset.target_names)

st.write(""" **Accuracy Of The Model** """)
model.score(X, Y)*100
prediction = model.predict(func)

st.write(""" **Prediction** """)
st.write(dataset.target_names[prediction])
