import numpy as np
import pandas as pd
import streamlit as st

# Function to learn specific and general hypotheses
def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    # Remove fully generalized hypotheses
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h

# Main function to create the Streamlit app
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Candidate Elimination Algorithm", page_icon="🧠")

    # Title and description
    st.title('Candidate Elimination Algorithm')
    st.markdown("""
    The Candidate Elimination Algorithm is a machine learning algorithm used for concept learning.
    It iteratively updates the most specific hypothesis and the most general hypothesis based on training data.
    Upload your training data (CSV file) to see the final specific and general hypotheses.
    """)

    # File uploader for CSV
    st.write("Upload your training data (CSV file):")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    # Process uploaded file
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display training data
        st.write("Training Data:")
        st.write(data)

        # Learn specific and general hypotheses
        concepts = np.array(data.iloc[:, 0:-1])
        target = np.array(data.iloc[:, -1])
        s_final, g_final = learn(concepts, target)

        # Display final hypotheses
        st.write("Final Specific_h:")
        st.write(s_final)

        st.write("Final General_h:")
        st.write(g_final)

        # Additional visual enhancements
        st.success("Model trained successfully!")
        st.markdown("---")
        st.info("Explore the final hypotheses and make predictions.")

# Run the app
if __name__ == "__main__":
    main()
