import streamlit as st
import pandas as pd

st.title("CSV File Reader")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.write("No file uploaded.")
else:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("CSV Preview (first 5 rows)")
        st.dataframe(df.head())
        
        # Optional: show as plain text like your Gradio version
        st.subheader("Plain Text Preview")
        st.text(df.head().to_string(index=False))

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")