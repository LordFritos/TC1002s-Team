import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import math

colour=['b','green','orange','cyan','magenta','k','y']

def distance(cluster):
	k=0
	run = 0
	while k < len(cluster):
		for i in range(run,len(cluster)):
			if i != k:
				diffs=[]
				cubics=0
				for j in range(len(cluster[0])):
					diffs.append(cluster[k,j]-cluster[i,j])
				for j in diffs:
					cubics+=j**2
				dist = math.sqrt(cubics)
				print("Distance between cluster {} and cluster {} is {} over {} dimensions\n".format(k,i,dist,len(cluster[0])))
		if run == 1:
			run+=1
		run+=1
		k+=1

def main():
	print('Se recomienda de 3 a 5 clusters')
	clusters = int(input('Enter number of clusters (max 7): '))
	data = pd.read_csv('./data/datasets_14701_19663_CC GENERAL.csv')
	npData = data.to_numpy()
	npData = npData[:, [2,7,8,9,10]] #2 & 7 original

	#KLEARN
	klearn = KMeans(n_clusters=clusters, random_state=0).fit(npData)
	#print(klearn.cluster_centers_)
	cols='{}'.format(data.columns[2])
	for c in range(len(data.columns)):
		if (c == 7 or c==8 or c==9 or c==10):
			cols+=',{}'.format(data.columns[c])
	print("\n\nIt ran {} iterations over {} clusters and {} dimensions ({})\n".format(klearn.n_iter_,clusters,len(npData[0,:]),cols))

	distance(klearn.cluster_centers_)

	#print(klearn.labels_)
	#print(len(npData),len(klearn.labels_))

	plt.title('KMeans learn')
	plt.xlabel('Balance Frequency')
	plt.ylabel('Purchase frequency')

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