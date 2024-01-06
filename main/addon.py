import streamlit as st




def newadd():
    styled_text = """
    <h3 style='color: black;'>Feedback</h3>
"""
    st.markdown(styled_text, unsafe_allow_html=True)

    form = st.form("text_input", clear_on_submit = True)

    with form:
        c1,c2,c3 = st.columns([5,1,5])
        with c1:
            name = st.text_input("Name")
        with c3:
            email = st.text_input("Email ID", key='email_id')

        feedback = st.text_area("Feedback about the application")

        method = st.text_input("Required Method")

        meth_desc = st.text_area("Method Description")

        ref = st.text_area("Reference")

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        file = open("feedback.txt", "a")

        file.write("Name: " + name + "\n")

        file.write("Feedback: " + feedback + "\n")

        file.write("Requested mehtod: " + method + "\n")

        file.write("Method description: " + meth_desc + "\n")

        file.write("Reference: " + ref + "\n")

        file.write("--------------------------------------------------------------------------------------------------" + "\n")

        



