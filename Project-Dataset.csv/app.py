import pandas as pd
import numpy as np
import pandas as pd
from collections import Counter
import itertools
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score

dataframe = pd.read_csv(r"C:\Users\MY PC\Downloads\Project-Dataset.csv\creditcard.csv.crdownload")
dataframe.head()

non_fraud = len(dataframe[dataframe.Class == 0])
fraud = len(dataframe[dataframe.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100


scaler = StandardScaler()
dataframe["NormalizedAmount"] = scaler.fit_transform(dataframe["Amount"].values.reshape(-1, 1))
dataframe.drop(["Amount", "Time"], inplace= True, axis= 1)
Y = dataframe["Class"]
X = dataframe.drop(["Class"], axis= 1)

(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size= 0.3, random_state= 42)
model = RandomForestClassifier(n_estimators= 100)

model.fit(train_X, train_Y)
# accuracy on training data
X_train_prediction = model.predict(train_X)
training_data_accuracy = accuracy_score(X_train_prediction, train_Y)*100
print('Accuracy on Training data : ', training_data_accuracy)

#web app
st.title("Credit card Fraud Detection Model")
input_df= st.text_input('Enter all required features values')
input_df_lst= input_df.split(",")

submit=st.button('Submit')

if submit:
    features = np.array(input_df_lst, dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))

    if prediction[0]==0:
        st.write('Legitimate transaction')
    else:
        st.write('Fraudulant transaction')

