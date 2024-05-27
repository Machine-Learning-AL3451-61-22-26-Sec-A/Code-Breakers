import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the Iris dataset
dataset = load_iris()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Initialize and train the KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

# Streamlit app title
st.title("Iris Dataset KNN Classifier")

# Display the dataset
st.write("### Iris Dataset")
iris_df = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
iris_df['target'] = dataset["target"]
st.write(iris_df.head())

# Function to display predictions
def display_predictions(X_test, y_test, kn):
    results = []
    for i in range(len(X_test)):
        x = X_test[i]
        x_new = np.array([x])
        prediction = kn.predict(x_new)
        result = {
            "Target": y_test[i],
            "Target Name": dataset["target_names"][y_test[i]],
            "Predicted": prediction[0],
            "Predicted Name": dataset["target_names"][prediction][0]
        }
        results.append(result)
    return pd.DataFrame(results)

# Display predictions
st.write("### Predictions")
predictions_df = display_predictions(X_test, y_test, kn)
st.write(predictions_df)

# Display accuracy
accuracy = kn.score(X_test, y_test)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
