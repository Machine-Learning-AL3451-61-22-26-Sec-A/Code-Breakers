import streamlit as st
import numpy as np
import pandas as pd

# Function to learn from data
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

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h

# Main function to run the Streamlit app
def main():
    st.title("""Candidate Elimination Algorithm 
    Team - NEURAL_KNIGHTS
    Contributors - Jeeva v,Dharaneesh,Jemson """)

    # Loading Data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.write(data)

        # Separating features and target
        concepts = np.array(data.iloc[:, :-1])
        target = np.array(data.iloc[:, -1])

        # Learning
        s_final, g_final = learn(concepts, target)

        # Displaying results
        st.write("\nFinal Specific_h:")
        st.write(s_final)
        st.write("\nFinal General_h:")
        st.write(g_final)

if __name__ == "__main__":
    main()
