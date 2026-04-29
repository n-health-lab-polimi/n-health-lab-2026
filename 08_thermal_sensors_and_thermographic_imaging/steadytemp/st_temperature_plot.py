import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("CSV Reader and Plotter")

# Initialize session state (Streamlit's equivalent of your global cache)
if "df" not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.write("No file uploaded.")
else:
    try:
        # Read CSV and store in session state
        df = pd.read_csv(uploaded_file)

        # Add readings column
        df["readings"] = np.arange(1, len(df) + 1)

        # Save to session state
        st.session_state.df = df

        # Show preview
        st.subheader("CSV Preview (first 5 rows)")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")

# If dataframe is loaded, show controls
if st.session_state.df is not None:
    df = st.session_state.df
    columns = list(df.columns)

    st.subheader("Plot Settings")

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X-axis column", columns)
    with col2:
        y_col = st.selectbox("Y-axis column", columns)

    # Plot
    if x_col and y_col:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df[x_col], df[y_col], marker='o')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Plot of {y_col} vs {x_col}")
            ax.grid(True)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")