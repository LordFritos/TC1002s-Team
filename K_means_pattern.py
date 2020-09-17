import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def main():
	data = pd.read_csv('./data/twitter_data/lego/table05262020_with_sentiment.csv')
	print(type(data))
	#plt.plot(data[:,13], data[:,14],data[:,15])


if __name__ == '__main__':
	main()