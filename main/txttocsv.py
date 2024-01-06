import csv
from io import StringIO
import os.path
import streamlit as st

def text_to_csv_converter():
    try:
        styled_text = """
        <h3 style='color: black;'>CSV Converter</h3>
    """
        st.markdown(styled_text, unsafe_allow_html=True)

        form = st.form("text_input", clear_on_submit = True)

        with form:
            csv_string = st.text_area('Enter the comma-separated dataset:')

            rows = csv_string.split('\n')

            csv_file = StringIO()
            writer = csv.writer(csv_file)
            for row in rows:
                writer.writerow(row.split(','))

            csv_output = csv_file.getvalue()

            filename = st.text_input('Enter a filename [with .csv extension]')


            submit_button = st.form_submit_button(label='Convert to CSV')

        if submit_button and not os.path.isfile(filename):
            with open(filename, 'w') as f:
                f.write(csv_output)
            st.write(" ")
            st.success(f'Successfully saved CSV output to {filename}.')

    except Exception as e:
        st.warning("Provide a valid filename")


