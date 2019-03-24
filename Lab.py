import sys
import math as m
from numpy import *
import numpy as np
from sklearn.cluster import KMeans

def main():
	##########
	##TASK 1##
	
	
	##########
	##TASK 2##
	a = array([[154,157,157,157,150,150,170,170,175,190],[154,157,157,151,153,155,180,180,170,190],
		   [154,157,150,154,160,160,160,155,155,165],[157,157,148,148,148,160,150,155,155,165],
		   [100,102,104,157,142,180,170,165,10,20],[100,103,105,165,155,180,175,162,40,50],
			   [100,102,108,132,180,180,172,167,25,63],[18,28,48,12,13,20,5,15,30,40],
			   [15,36,46,18,21,22,28,32,30,36],[17,21,24,26,35,45,28,30,40,20]])
	
	b = array([[152,156,157,156,149,150,170,160,175,190],[154,159,157,151,153,155,180,180,170,190],
		   [153,157,155,154,160,160,160,155,155,165],[157,157,148,148,148,160,150,155,155,165],
		   [101,102,104,159,143,180,170,165,110,220],[99,103,105,164,155,179,175,162,240,250],
			   [100,102,108,132,180,180,172,167,155,163],[118,123,148,129,109,120,155,215,140,180],
			   [156,136,210,218,175,122,128,232,180,156],[178,231,245,226,215,145,188,230,170,140]])
	
	im = np.absolute(np.array(a) - np.array(b))
	
	print(a)
	print("\n")
	print(b)
	print("\n")
	print(im)
	print("\n")
	
	im_max, im_min = im.max(), im.min()
	im = (im - im_min)/(im_max - im_min)
	print("After normalization:")
	print(im)
	new_im = []
	for i in range(len(im)):
		for j in range(len(im)):
			new_im.append(im[i][j])
	new_im = np.array(new_im)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(new_im.reshape(-1, 1))
	labels = kmeans.labels_
	print("After clustering: ")
	print(labels.reshape(10,10))

	##########
	##Task 3##
	
	cryptoCurPrices = np.array([7845, 778, 942, 143, 0.75, 7956, 810, 976, 146, 0.76, 8215, 825, 1002, 152,
					   0.78, 8542, 847, 1038, 157, 0.78, 8150, 100587, 807, 1015, 150, 0.72, 8386,
					   884, 101964, 1085, 138, 0.82, 8219, 827, 995, 158, 0.82, 7500, 745, 948,
					   135, 0.67, 9257, 901, 120967, 1154, 148, 0.72, 8553, 811, 1218, 175, 0.84])

	cryptoCurPricesArr = [7845, 778, 942, 143, 0.75, 7956, 810, 976, 146, 0.76, 8215, 825, 1002, 152,
					   0.78, 8542, 847, 1038, 157, 0.78, 8150, 100587, 807, 1015, 150, 0.72, 8386,
					   884, 101964, 1085, 138, 0.82, 8219, 827, 995, 158, 0.82, 7500, 745, 948,
					   135, 0.67, 9257, 901, 120967, 1154, 148, 0.72, 8553, 811, 1218, 175, 0.84]
	Corrupt = []
	MinVal = []
	MaxVal = []
	for val in cryptoCurPrices:
		if val < 0.0 or val > 20000:
			Corrupt.append(val)
		if val > 5000 and val <= 20000:
			MaxVal.append(val)
		if val >= 0.0 and val <= 5000:
			MinVal.append(val)
	Corrupt.sort()
	MinVal.sort()
	MaxVal.sort(reverse=True)
	print("\nCorrupted values: ")
	print(Corrupt)
	print("Minimum values: ")
	print(MinVal)
	print("Maximum values: ")
	print(MaxVal)
	print("\n")
	#print (cryptoCurPrices)
	#print (cryptoCurPricesArr)
	cryptoCurPrices.sort()
	cryptoCurPricesArr.sort()
	kmeans = KMeans(n_clusters=4, random_state=0).fit(cryptoCurPrices.reshape(-1,1))
	labels = kmeans.labels_
	
	for i in range(len(cryptoCurPricesArr)):
		print(cryptoCurPricesArr[i], labels[i])

main()