#!/usr/bin/env python
# coding: utf-8

# In[106]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make_blobs dataset for clustering.
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=3, random_state=42)
# Number of training examples and cluster centers.
m = X.shape[0] 
n = X.shape[1] 
n_iter = 50


# In[107]:


# Plot the clusters.
plt.figure(figsize=(10, 7), dpi=100)
plt.scatter(X[:,0],X[:,1])
plt.title('Original Dataset')


# In[108]:


# Shape of the dataset.
print(X.shape, y.shape)
print(m,n)


# In[130]:


# Compute the initial centroids randomly.
import random
K=3
# Create an empty centroid array.
centroids = np.array([]).reshape(0,n) #shape = (0,2)
# Create 3 random centroids.
for k in range(K):
    centroids = np.r_[centroids, X[random.randint(0,m-1)].reshape(1,2)] #randint(0,249)

"""
>>>V = array([1,2,3,4,5,6 ])
>>>Y = array([7,8,9,10,11,12])
>>>np.r_[V[0:2],Y[0],V[3],Y[1:3],V[4:],Y[4:]]
array([ 1,  2,  7,  4,  8,  9,  5,  6, 11, 12])
"""


# In[131]:


# Plot the centroids.
plt.figure(figsize=(10, 7), dpi=100)
plt.scatter(centroids[:,0], centroids[:,1])
plt.title('Centroids')


# In[123]:


def euclid(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)


# In[124]:


def plot(m,X,label,centroids):
    plt.figure(figsize=(10, 7), dpi=100)
    color = ['red','green','blue']
    for i in range(m):
        plt.scatter(X[i,0],X[i,1], color = color[label[i]])
    for k in range(K):
        plt.scatter(centroids[k,0],centroids[k,1], color = "purple")
    plt.figure(figsize=(10, 7), dpi=100)
    plt.show()


# In[132]:


for it in range(0,51):
    label = []
    
    for i in range(m):
        distance = []
        for j in range (K):
            dist = euclid(X[i],centroids[j])
            distance.append(dist)
            
        min_in = np.argmin(distance)
        label.append(min_in)
    label = np.array(label)
    prev_cent = centroids.copy()
        
    #Update the centroids
    for k in range(K):
        k_in = np.where(label==k)
        mean = np.mean(X[k_in], axis = 0)
        centroids[k] = mean
        
    if(it%4==0):
        plot(m,X,label,centroids)
        
    #converge = []
        
    #for k in range(K):
        #conv = abs(prev_cent[k]-centroids[k])
        #converge.append(conv)
        
    #converge= np.array(converge)
    converge = [np.abs(prev_cent-centroids) for k in range(K)]
    if(np.sum(converge)<0.3):
        print(f"Converged in iteration {it}")
        plot(m,X,label,centroids)
        break
    
    


# In[129]:


# Repeat the above steps 
for it in range(n_iter):
    label = []
    # Assign clusters to points.
    for i in range(m):
        distance = [euclid(X[i],centroids[k]) for k in range(K)]
        min_index = np.argmin(distance)
        label.append(min_index)
    label = np.array(label)
    old_centroids = centroids.copy()
    
    # Compute mean and update.
    for k in range(K):
        k_index = np.where(label==k)
        mean_centroid = np.mean(X[k_index],axis=0)
        centroids[k] = mean_centroid
    #plot graph in every 5 iterations
    if(it%4==0):
        plot(m,X,label,centroids)
    convergence = [np.abs(old_centroids-centroids) for k in range(K)]
    if(np.sum(convergence)<0.5):
        print(f"Converged in iteration {it}")
        plot(m,X,label,centroids)
        break
# cost = np.sum([euclid_distance(X[i],centroid[label[i]]) for i in range(m)])/m


# In[ ]:




