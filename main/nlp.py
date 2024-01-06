import streamlit as st
import vanna as vn
from io import StringIO
import sys

@st.cache_data
def create():
    api_key = vn.get_api_key('sriram03p@gmail.com')  # Replace with your email
    vn.set_api_key(api_key)
    vn.set_model("chinook")
create()



def main():
    st.title("SQL Query App")

    # Get user input for SQL query
    sql_query = st.text_input("Enter your SQL query:")

    # Execute SQL query when the user clicks the "Generate SQL Query" button
    if st.button("Generate SQL Query"):
        try:
            


            # Redirect standard output to capture printed output
            original_stdout = sys.stdout
            sys.stdout = StringIO()


            # Use vanna to run the SQL query
            vn.ask(sql_query)

            # Capture printed output
            result = sys.stdout.getvalue()

            # Reset standard output
            sys.stdout = original_stdout

            # Display the generated SQL query result
            st.success("Generated SQL Query:")
            st.code(result, language='sql')
        except Exception as e:
            st.error(f"Error executing SQL query: {e}")

# Run the app
if __name__ == "__main__":
    main()