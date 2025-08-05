import joblib
import pandas as pd
import streamlit as st

input_file = None

st.title("Predict stroke")
st.write(
    "In this app, you can input patient data and a :blue[**Random Forest Classifier**] will \
predict if the patient has a high or low risk of stroke based on a sample of previous stroke patients."
)

input_type = st.selectbox(
    "Choose how you want to enter patient information:", ["Manually", "CSV"], index=None
)

if input_type == "Manually":
    st.write("#### Fill out the form")
    # inputs
    gender = st.selectbox("What is the gender?", ["Male", "Female"])
    age = st.number_input(
        "What is the age?", value=30, min_value=0, max_value=120, step=1
    )
    height = st.number_input(
        "What is the height (in cm)?", value=179, min_value=0, max_value=250, step=1
    )
    height = height / 100
    weight = st.number_input(
        "What is the weight (in KG)?", value=80, min_value=0, step=1
    )
    hypertension = st.toggle("Hypertension")
    heart_disease = st.toggle("Heart Disease")
    ever_married = st.toggle("Was he/she ever married (current or divorced?")
    work_type = st.selectbox(
        "What is the worktype?",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    )
    residence_type = st.selectbox("WHat is the residence type?", ["Urban", "Rural"])
    avg_glucose_level = st.number_input(
        "What is the average glucose level (in mg/dL)?", value=106.14
    )
    smoking_status = st.selectbox(
        "What is the current smoking status?",
        ["formerly smoked", "never smoked", "smokes", "Unknown"],
    )
    if height != 0:
        bmi = weight / height**2
    else:
        bmi = 0

    # Transform inputs into pd.DataFrame
    features = {
        "gender": [gender],
        "age": [age],
        "height": [height],
        "weight": [weight],
        "hypertension": [int(hypertension)],
        "heart_disease": [int(heart_disease)],
        "ever_married": [int(ever_married)],
        "work_type": [work_type],
        "residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "smoking_status": [smoking_status],
        "bmi": [bmi],
    }

    input_file = pd.DataFrame(features)
    st.write(input_file)

elif input_type == "CSV":
    file = st.file_uploader("Upload CSV file:", type=["CSV"])

    # Transform file into pd.DataFrame
    if file is not None:
        input_file = pd.read_csv(file)
        og_file = st.toggle("Processed file / Original file")
        if og_file:
            st.write(input_file)
        elif not og_file:
            input_file = input_file.drop(["id", "address", "stroke"], axis=1).assign(
                bmi=lambda df: df["weight"] / df["height"] ** 2
            )
            st.write(input_file)
else:
    pass

# import model
pipeline = joblib.load("ml-model.pkl")


# Outcome
if input_type is not None:
    if input_file is not None:
        button = st.button("Predict")

        if button:
            res = pipeline.predict(input_file)
            if res.shape[0] == 1:
                if res[0]:
                    output = ":red[**HIGH**]"
                else:
                    output = ":green[**LOW**]"
                st.write("#### Outcome")
                st.write(f"Prediction: :blue[**{bool(res[0])}**].")
                st.write(f"The patient has a {output} risk of stroke.")
            else:
                st.write(pd.DataFrame({"stroke": res}))
