import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2})
    return data

data = load_data()

# Features & Target
X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = data["Survived"]

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


st.title("Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival chance.")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Fare (Ticket Price)", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert categorical inputs
sex_val = 0 if sex == "Male" else 1
embarked_val = {"C": 0, "Q": 1, "S": 2}[embarked]

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f" Passenger would SURVIVE with probability {probability:.2f}")
    else:
        st.error(f" Passenger would NOT survive (Survival chance: {probability:.2f})")

