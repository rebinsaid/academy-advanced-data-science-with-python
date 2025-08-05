import streamlit as st
import pandas as pd
import numpy as np


def main():
    model = load('models/name.joblib')
    data_df = pd.read_csv('data/data_df.csv')
    image = "data/image.png"

    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

    st.title("Welcome to the ------ Prediction App!")
    st.image(image, use_column_width=True)
    st.header("Header?")
    st.subheader("Subheader")

    #user inputs
    text = st.text_input("What is the relevant text?", "")
    slider_var = st.slider("How large should this slider's value be?", 0, 200, 0, 1)
    date = st.date_input('Which date is it?', value = pd.Timestamp('2020-03-14'))
    num_var = st.number_input("Choose a number! ", 10, 500, 250, step = 10)

    if st.button("Get prediction"):

        prediction = predict_func(X_input, model)
            
if __name__ == "__main__":
    main()


 