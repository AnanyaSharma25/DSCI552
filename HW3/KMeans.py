#importing all modules and libraries needed 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from pandas import DataFrame
from sklearn.cluster import KMeans

#load patients data
data = np.load('/Users/ananyasharma/Downloads/ps3_genetic_fingerprints.npy')

#check dataset shape to know rows and columns
data.shape

#now find optimum k. For this Kmeans would be run for k=3 till when the decrease in the model's inertia
#becomes linear
#inertia is the sum of squared error which basically is used as a measure of variance

#for k=3 (i.e. 3 clusters)
kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(data)

#KMeans algorithm clusters data by separating samples into n groups, minimizing inertia.
#KMeans algo. divides N sample into K disjoint clusters C.

#inertia of the model with 3 clusters
kmeans3.inertia_

#for k=4
kmeans4 = KMeans(n_clusters=4)
y_kmeans4 = kmeans4.fit_predict(data)

#inertia of the model with 4 clusters
kmeans4.inertia_


#for k=5 (i.e. 5 clusters)
kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(data)

#inertia of the model with 5 clusters
kmeans5.inertia_


#for k=6 (i.e. 6 clusters)
kmeans6 = KMeans(n_clusters=6)
y_kmeans6 = kmeans6.fit_predict(data)

#inertia of the model with 6 clusters
kmeans6.inertia_


#for k=7 (i.e. 7 clusters)
kmeans7 = KMeans(n_clusters=7)
y_kmeans7 = kmeans7.fit_predict(data)

#inertia of the model with 7 clusters
kmeans7.inertia_


#for k=8 (i.e. 8 clusters)
kmeans8 = KMeans(n_clusters=8)
y_kmeans8 = kmeans8.fit_predict(data)

#inertia of the model with 8 clusters
kmeans8.inertia_


#for k=9 (i.e. 9 clusters)
kmeans9 = KMeans(n_clusters=9)
y_kmeans9 = kmeans9.fit_predict(data)

#inertia of the model with 9 clusters
kmeans9.inertia_


#compiling all to loop through so that the values can be visualised
#n_init (default value 10): Represents the number of time the k-means algorithm will be run independently, with different random centroids in order to choose the final model as the one with the lowest SSE.
#init (default as k-means++): Represents method for initialization. The default value of k-means++ represents the selection of the initial cluster centers (centroids) in a smart manner to speed up the convergence. The other values of init can be random, which represents the selection of n_clusters observations at random from data for the initial centroids
#random was chosen as k-means++ is a complexer algo not used here.
#max_iter (default value 300): Represents the maximum number of iterations for each run. The iteration stops after the maximum number of iterations is reached even if the convergence criteria is not satisfied. This number must be between 1 and 999.

kmeans_kwargs = {
       "init": "random",
        "n_init": 10,
        "max_iter": 300
    }

# A list holds the SSE/inertia values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)
    
#the graph plots the number of clusters against the inertia of the models with that number of clusters
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE / Inertia")
plt.show()


#before using PCA data has to be scaled

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import pandas as pd

# normalizing the data
#Standardization scales/shifts the values for each numerical feature so that the features have a mean of 0 and standard deviation of 1
std_data = StandardScaler().fit_transform(data)  

#fitting the data
pca = PCA()
pca.fit(std_data)

#this graph plots the number of components against their variance ratio.
#this graph is used to see how many components of a dataset contribute to its variance and should be selected as Principal Components to reduce the same.
plt.figure(figsize=(10,10))
plt.plot(range(0,512),pca.explained_variance_ratio_.cumsum(),marker='o',linestyle='--')

plt.title('Explained Variance by Features')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance')


#Finding out which component contributes most to variance
#since we can't select 200 components, lets start with 20 and see PCA variance on them

pca = PCA(n_components=20)
principalComponents = pca.fit_transform(std_data)

# Plotting the variances for each PC
PC = range(pca.n_components_)
plt.bar(PC, pca.explained_variance_ratio_, color='gold')
plt.xlabel('Principal Components')
plt.ylabel('Variance %')
plt.xticks(PC)

# Putting components in a dataframe for later
PCA_components = pd.DataFrame(principalComponents)

#Visualising the clusters after choosing 2 PCs out of 3 
model = KMeans(n_clusters=7)
model.fit(PCA_components.iloc[:,:2])

labels = model.predict(PCA_components.iloc[:,:2])

plt.scatter(PCA_components[0], PCA_components[1], c=labels)
plt.show()

#Cluster Centers
model.cluster_centers_

#Counting the number of datapoints in each cluster
from collections import Counter, defaultdict
print(Counter(model.labels_))

#loading Patient Z's data and making one single data array combining the general patient and patient z's list
data1 = np.load('/Users/ananyasharma/Downloads/ps3_patient_zet.npy')
x = np.vstack(data1)
z = np.concatenate((data, x.T), axis=0)
z.shape

#model is trained again with 7 clusters with scaled data to see which cluster patient Z belongs to 
std_data1 = StandardScaler().fit_transform(z) 
pca = PCA()
pca.fit(std_data1)

pca = PCA(n_components=20)
principalComponents = pca.fit_transform(std_data)

# Plotting the variances for each PC
PC = range(pca.n_components_)
plt.bar(PC, pca.explained_variance_ratio_, color='gold')
plt.xlabel('Principal Components')
plt.ylabel('Variance %')
plt.xticks(PC)

# Putting components in a dataframe for later
PCA_components = pd.DataFrame(principalComponents)

model1 = KMeans(n_clusters=7)
model1.fit(PCA_components.iloc[:,:2])
labels = model1.predict(PCA_components.iloc[:,:2])
plt.scatter(PCA_components[0], PCA_components[1], c=labels)
plt.show()

#Counting the number of datapoints in each cluster
from collections import Counter, defaultdict
print(Counter(model1.labels_))

