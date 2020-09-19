import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

colour=['b','green','orange','cyan','magenta','k','y']

def main():
	data = pd.read_csv('./data/datasets_14701_19663_CC GENERAL.csv')
	npData = data.to_numpy()
	"""
	npData = np.delete(npData,obj=0, axis=1)
	npData = np.delete(npData,obj=0, axis=1)
	npData = np.delete(npData,obj=1, axis=1)
	npData = np.delete(npData,obj=1, axis=1)
	npData = np.delete(npData,obj=1, axis=1)
	"""
	npData = npData[:, [2,7,8,9,10]] #2 & 7 original

	#KLEARN
	klearn = KMeans(n_clusters=4, random_state=0).fit(npData)
	#print(klearn.cluster_centers_)
	print("It ran {} iterations".format(klearn.n_iter_))
	#print(klearn.labels_)
	#print(len(npData),len(klearn.labels_))

	plt.title('KMeans learn')
	plt.xlabel('Balance Frequency') #CHANGE
	plt.ylabel('Purchase frequency')
	print(npData[:,0])
	print(npData[:,1])

	#Cluster data
	for i in range(len(klearn.labels_)):
		X = npData[i,0]
		Y = npData[i,1]
		plt.scatter(X,Y,color=colour[klearn.labels_[i]])

	#cluster centers
	x = klearn.cluster_centers_[:,0]
	y = klearn.cluster_centers_[:,1]
	plt.scatter(x,y,color='r',marker='+',s=100)
	plt.show()


if __name__ == '__main__':
	main()