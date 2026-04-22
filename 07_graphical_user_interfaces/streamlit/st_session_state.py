import streamlit as st

st.title("Streamlit sessions")


if "counter" not in st.session_state:
    st.session_state.counter = 0
    #st.session_state["counter"] = 0

increment = st.button('Increment')
if increment:
    st.session_state.counter += 1

st.write('Count = ', st.session_state.counter)

