import pickle
import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB

# Load the pre-trained Naive Bayes model and scaler
model_nb = pickle.load(open("model/model_nb.pkl", 'rb'))
scaler = pickle.load(open("model/scaler.pkl", 'rb'))

# Load the dataset
df = pd.read_csv('dataset/diabetes_prediction_dataset.csv')

# Preprocess the data
df = df.drop(df[df['gender'] == 'Other'].index)
df = df.drop_duplicates()
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['smoking_history'] = df['smoking_history'].map({'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5})

# Handle Imbalanced Data
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(df.drop(labels="diabetes", axis=1).values, df["diabetes"].values)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.10, random_state=4)

# Standardization
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model_nb.fit(X_train_scaled, y_train)

# Streamlit App
st.title('Diabetes Prediction App')

# Sidebar for user input
st.sidebar.title('User Input')

# Collect user input
jenis_kelamin = st.sidebar.selectbox('Gender', ['Female', 'Male'])
umur = st.sidebar.number_input('Age', min_value=1, max_value=100, value=25)
hipertensi = st.sidebar.radio('Hypertension', ['never', 'ever'], index=1)
penyakit_jantung = st.sidebar.radio('Heart Disease', ['never', 'ever'], index=0)
bmi = st.sidebar.number_input('BMI', min_value=1.0, max_value=50.0, value=25.0)
darah = st.sidebar.number_input('Cholesterol', min_value=1.0, max_value=500.0, value=200.0)
gula_darah = st.sidebar.number_input('Blood Sugar Level', min_value=50, max_value=300, value=70)
status_merokok = st.sidebar.selectbox('Smoking History', ['never', 'No Info', 'current', 'former', 'ever', 'not current'], index=0)

jenis_kelamin = 0 if jenis_kelamin == 'Female' else 1
hipertensi = 1 if hipertensi == 'ever' else 0
penyakit_jantung = 1 if penyakit_jantung == 'ever' else 0

status_merokok_mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
status_merokok = status_merokok_mapping[status_merokok]

# Standardize the user input
user_input = np.array([[jenis_kelamin, umur, hipertensi, penyakit_jantung, bmi, darah, gula_darah, status_merokok]])
user_input_scaled = scaler.transform(user_input)

# # Make prediction
# prediction = model_nb.predict(user_input_scaled)
# Make prediction when user input changes
if st.sidebar.button('Predict'):
    # Make prediction
    prediction = model_nb.predict(user_input_scaled)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()
    
    # Display prediction
    
    if prediction == 0:
      result = ":green[**Tidak Diabetes**]"
    else:
      result = ":red[**Diabetes**]"

    st.write("")
    st.write("")
    st.write('**Prediction Result:**')
    st.subheader(result)