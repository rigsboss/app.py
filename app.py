{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "75fee992",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b0dcd4d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('Data\\student-mat.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "178c3960",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Title for the app\n",
    "st.title('Secondary School Student Performance Prediction')\n",
    "\n",
    "#input fields for features\n",
    "feature1 = st.slider('G1', min_value=0, max_value=100, value=50)\n",
    "feature2 = st.slider('G2', min_value=0, max_value=100, value=50)\n",
    "\n",
    "#button to trigger prediction\n",
    "if st.button('Predict'):\n",
    "    # Perform prediction using the selected features\n",
    "    X= df.drop ('G3', axis = 1)# Features\n",
    "    y = df.G3  # Target variable\n",
    "\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "    lr = LinearRegression()\n",
    "    lr.fit(X_train, y_train)\n",
    "\n",
    "    knn = KNeighborsRegressor(n_neighbors=5)\n",
    "    knn.fit(X_train, y_train)\n",
    "\n",
    "    features = np.array([[feature1, feature2]])  # Use the selected features for prediction\n",
    "    y_pred_lr = lr.predict(features)\n",
    "    y_pred_knn = knn.predict(features)\n",
    "\n",
    "    # Display the prediction results\n",
    "    st.write('Linear Regression Prediction:', y_pred_lr[0])\n",
    "    st.write('K-Nearest Neighbors Prediction:', y_pred_knn[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "df60ebdc",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (Temp/ipykernel_14624/4054717884.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"C:\\Users\\david\\AppData\\Local\\Temp/ipykernel_14624/4054717884.py\"\u001b[1;36m, line \u001b[1;32m1\u001b[0m\n\u001b[1;33m    streamlit run your_script.py\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "streamlit run your_script.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a162cda",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
