import streamlit as st
import numpy as np
import pandas as pd
from numpy.random import default_rng as rng

st.text_input("Your name", key="name")

st.subheader("Columns example")
col1, col2 = st.columns(2)

with col1:
    x = st.slider("Select a value", 1, 10)
with col2:
    st.write("The value of :red[***x***] is", x)


# Plotting data of a dataframe
chart_data = pd.DataFrame(
    np.random.randn(20, 3), 
    columns=["a", "b", "c"])


st.line_chart(chart_data)


# Adding a checkbox
if st.checkbox("Show dataframe"):
    chart_data = pd.DataFrame(
        np.random.randn(20, 3), 
        columns=("a", "b", "c"))
    chart_data


 
# Creating a dataframe    
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

# Adding a dropdown menu
option = st.selectbox(
    "Which number do you like best?", 
    df['first column'])

"You selected: ", option


# Adding a sidebar (on the left)
add_selectbpox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

add_slider = st.sidebar.slider(
    "Select a range of values",
    0.0, 100.0, (25.0, 75.0)
)


left_column, right_column = st.columns(2)
left_column.button("Press me!")

with right_column:
    chosen = st.radio(
        "Sort hat",
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")


    



