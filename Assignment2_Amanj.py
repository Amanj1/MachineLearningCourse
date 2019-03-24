import sys
import math as m
from numpy import *
import numpy as np
from sklearn.cluster import KMeans

def step(x1,x2,w1,w2,bias):
	val = x1*w1+x2*w2+bias
	if val >= 0:
		return 1
	return 0

def perceptron_neural_2input(X1,X2,Yd,w1,w2,Lr,Tb):
	weight1 = w1
	weight2 = w2
	Learning_rate = Lr
	Threshold_bias = Tb
	tmp_error = [0]* len(X1)
	num_it = 1
	Run = 1
	check = 0
	while Run > 0:
		for i in range(len(X1)):
			y = step(X1[i],X2[i],weight1,weight2,Threshold_bias)
			tmp_error[i] = Yd[i] - y
			weight1 = weight1 + Learning_rate*X1[i]*tmp_error[i]
			weight2 = weight2 + Learning_rate*X2[i]*tmp_error[i]
			Threshold_bias = Threshold_bias + Learning_rate*Threshold_bias*tmp_error[i]
		check = 0
		for e in tmp_error:
			if e != 0:
				check = 1
				num_it = num_it + 1
		if check == 0:
			Run = 0
	r = [weight1, weight2, num_it]
	return r

def main():
	##########
	##TASK 1##
	# var1  var2	OR
	# 0		0		0
	# 0		1		1
	# 1		0		1
	# 1		1		1
	#
	 
	InputVariable1 = [0,0,1,1]
	InputVariable2 = [0,1,0,1]
	ORoperation = [0,1,1,1] ## Results based on input values from variable 1 and 2
	result = perceptron_neural_2input(InputVariable1,InputVariable2,ORoperation,0.5,0.1,0.01,-0.1)
	print("Values from assignment 2, Task 1")
	print("Weight 1: ", result[0]," Weight 2: ", result[1], " number of iteration: ", result[2])
	
	inputVar1 = [0,0,1,1]
	inputVar2 = [0,1,0,1]
	desOutput = [0,0,0,1]
	
	print("\nTesting different dataset")
	result = perceptron_neural_2input(inputVar1,inputVar2,desOutput,0.3,-0.1,0.1,-0.2)
	print("Weight 1: ", result[0]," Weight 2: ", result[1], " number of iteration: ", result[2])
	
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
	
	print("\nTask 2, first matrice is a, second is b and third is absolute difference between the matrices\n")
	print(a)
	print("\n")
	print(b)
	print("\n")
	print(im)
	print("\n")
	
	im_max, im_min = im.max(), im.min()
	im = (im - im_min)/(im_max - im_min)
	print("After normalization of the 3rd matrice:")
	print(im)
	new_im = []
	for i in range(len(im)):
		for j in range(len(im)):
			new_im.append(im[i][j])
	new_im = np.array(new_im)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(new_im.reshape(-1, 1))
	labels = kmeans.labels_
	print("\nAfter clustering the normalized matrice with 2 clusters. 0 is the unchanged and 1 is the changed values: ")
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
	print("\nTask 3")
	print("\nCorrupted values that are less than 0 or greater than 20000: ")
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
	print("Using 4 clusters to classify the crypto currency prices.")
	kmeans = KMeans(n_clusters=4, random_state=0).fit(cryptoCurPrices.reshape(-1,1))
	labels = kmeans.labels_
	
	for i in range(len(cryptoCurPricesArr)):
		print("Price:", cryptoCurPricesArr[i], " Cluster:" ,labels[i])
main()