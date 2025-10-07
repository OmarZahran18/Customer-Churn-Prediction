import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Testing Data
data = {
    'Age': [25, 30, 35, 40, 50, 60, 23, 45, 33, 55],
    'Tenure': [1, 2, 5, 3, 8, 10, 2, 6, 4, 7],
    'Gender': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],  # 0=Male, 1=Female
    'Churn': [0, 0, 1, 0, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Age', 'Tenure', 'Gender']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

st.markdown("<h2 style='color:blue;'>Customer Churn Prediction</h2>", unsafe_allow_html=True)
st.write("Enter customer details to predict if they are likely to churn or stay.")

st.markdown("### Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Average Age", f"{df['Age'].mean():.1f}")
col2.metric("Average Tenure (years)", f"{df['Tenure'].mean():.1f}")
col3.metric("Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")

st.sidebar.header("Input Customer Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, help="Enter customer's age")
tenure = st.sidebar.number_input("Tenure (years)", min_value=0, max_value=10, value=3, help="How many years customer stayed")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender_num = 0 if gender == "Male" else 1

threshold = st.sidebar.slider("Churn Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Set probability threshold to decide churn")

if 'predictions_log' not in st.session_state:
    st.session_state['predictions_log'] = []

if st.button("Predict"):
    with st.spinner('Predicting...'):
        input_data = [[age, tenure, gender_num]]
        proba = model.predict_proba(input_data)[0][1] 
        prediction = 1 if proba >= threshold else 0
        
    st.write(f"Churn Probability: {proba*100:.2f}%")
    
    if prediction == 1:
        st.error("The customer is likely to CHURN.")
    else:
        st.success("The customer is likely to STAY.")
    
    fig, ax = plt.subplots()
    ax.bar(["Stay", "Churn"], [100 - proba*100, proba*100], color=['green', 'red'])
    ax.set_ylabel('Probability (%)')
    ax.set_ylim([0, 100])
    st.pyplot(fig)
    
    st.session_state.predictions_log.append({
        "Age": age,
        "Tenure": tenure,
        "Gender": gender,
        "Churn Probability": f"{proba*100:.2f}%",
        "Prediction": "CHURN" if prediction == 1 else "STAY"
    })

if st.session_state.predictions_log:
    st.markdown("### Prediction History")
    history_df = pd.DataFrame(st.session_state.predictions_log)
    st.dataframe(history_df)
