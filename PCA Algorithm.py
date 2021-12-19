
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 10:49:11 2021

@author: Sinem
"""
#PCA Algorithm
#The code for split the iris dataset into 2 components with PCA algorithm
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#pull iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#load dataset into pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
#first 5 data
print(df.head())

#features
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

#seperate features and target
X = df.loc[:, features].values
y = df.loc[:,['target']].values
#The first step of the PCA algorithm is standardization
X = StandardScaler().fit_transform(X)


#I can split it into as many component as i want, im reducing 4 features to 2 features
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['component1', 'component2'])
#listing features as reduced in size
print(principalDf.head())

#I'm adding the target equivalent, reduced from 4 features to 2 features
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf.head())       


#Visualizing the PCA algorithm with rgb values ​​(we can see the size reduced in the 2 component distribution)
# We can see in 2D component1 and component2
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Component1', fontsize=17)
ax.set_ylabel('Component2', fontsize=17)
ax.set_title('2 Component PCA', fontsize=25)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#rgb
colors = ['r','g','b']

for target, color in zip(targets,colors):
 indicesToKeep = finalDf['target'] == target
 ax.scatter(finalDf.loc[indicesToKeep, 'component1'], 
finalDf.loc[indicesToKeep, 'component2'], c = color, s = 50)
    
ax.legend(targets)
print(ax.grid())


         