from numba import jit
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def main():
	data = pd.read_csv('./data/iris.data')
	npData = data.to_numpy()
	print(npData)
	X = np.array([f[:-1] for f in npData])
	Y= npData[-1]
	plt.plot(X,Y)
	plt.show()
	#kmeans = KMeans(n_clusters= 3)


if __name__ == '__main__':
	main()