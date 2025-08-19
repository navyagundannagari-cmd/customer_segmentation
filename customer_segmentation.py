
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('customers.csv')

# Basic EDA
print(df.head())
print(df.describe())

# Visualizing distributions
sns.displot(df['Annual Income (k$)'], kde=True)
plt.title('Distribution of Annual Income')
plt.show()

sns.displot(df['Spending Score (1-100)'], kde=True)
plt.title('Distribution of Spending Score')
plt.show()

# Elbow method to find optimal number of clusters
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Apply KMeans with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segments')
plt.show()
