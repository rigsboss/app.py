import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('Data\student-mat.csv')
#Title for the app
st.title('Secondary School Student Performance Prediction')

#input fields for features
feature1 = st.slider('G1', min_value=0, max_value=100, value=50)
feature2 = st.slider('G2', min_value=0, max_value=100, value=50)

#button to trigger prediction
if st.button('Predict'):
    # Perform prediction using the selected features
    X= df.drop ('G3', axis = 1)# Features
    y = df.G3  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    features = np.array([[feature1, feature2]])  # Use the selected features for prediction
    y_pred_lr = lr.predict(features)
    y_pred_knn = knn.predict(features)

    # Display the prediction results
    st.write('Linear Regression Prediction:', y_pred_lr[0])
    st.write('K-Nearest Neighbors Prediction:', y_pred_knn[0])