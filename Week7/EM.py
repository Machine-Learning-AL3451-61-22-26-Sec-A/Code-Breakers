import streamlit as st
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Iris Dataset Clustering: KMeans vs GMM")

# Load the Iris dataset
dataset = load_iris()

# Create DataFrame for features and target
X = pd.DataFrame(dataset.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(dataset.target, columns=['Targets'])

# Display the dataset
st.write("### Iris Dataset")
st.write(X.head())

# Function to create scatter plots
def plot_clusters(X, y, predY=None, title=""):
    colormap = np.array(['red', 'lime', 'black'])
    plt.figure(figsize=(5, 5))
    if predY is None:
        plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
    else:
        plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY], s=40)
    plt.title(title)
    st.pyplot(plt)

# Real Plot
st.write("### Real Classification")
plot_clusters(X, y, title='Real')

# KMeans Clustering
st.write("### KMeans Clustering")
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(X)
predY_kmeans = np.choose(kmeans_model.labels_, [0, 1, 2]).astype(np.int64)
plot_clusters(X, y, predY=predY_kmeans, title='KMeans')

# GMM Clustering
st.write("### GMM Clustering")
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
gmm_model = GaussianMixture(n_components=3)
gmm_model.fit(X_scaled)
predY_gmm = gmm_model.predict(X_scaled)
plot_clusters(X, y, predY=predY_gmm, title='GMM Classification')

# Show the plots side by side
st.write("### Side-by-Side Comparison")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
colormap = np.array(['red', 'lime', 'black'])

# Real Plot
axs[0].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
axs[0].set_title('Real')

# KMeans Plot
axs[1].scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY_kmeans], s=40)
axs[1].set_title('KMeans')

# GMM Plot
axs[2].scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY_gmm], s=40)
axs[2].set_title('GMM Classification')

for ax in axs:
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')

st.pyplot(fig)
