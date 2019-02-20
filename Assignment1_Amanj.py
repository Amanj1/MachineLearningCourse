import sys
import math as m

def mean(alist):
    return float(sum(alist)) / max(len(alist), 1)

def correlationCoeff(x,y):
	xMean = mean(x)
	yMean = mean(y)
	SxxSum = 0
	SyySum = 0
	SxySum = 0
	Sxx = 0
	Syy = 0
	Sxy = 0
	#Sxx
	for i in x:
		SxxSum = SxxSum + m.pow((i-xMean),2)
	Sxx = SxxSum / (len(x)-1)
	Sxx = m.sqrt(Sxx)
	#Syy
	for i in y:
		SyySum = SyySum + m.pow((i-yMean),2)
	Syy = SyySum / (len(y)-1)
	Syy = m.sqrt(Syy)
	#Sxy
	for i in range(len(y)):
		SxySum = SxySum + (x[i]-xMean)*(y[i]-yMean)
	Sxy = SxySum / (len(y)-1)
	r = Sxy/(Sxx*Syy)
	return r

def euclideanDistance(testData, trData):
	distance = 0
	for x in range(len(trData)):
		distance += pow((testData[x] - trData[x]), 2)
	return m.sqrt(distance)

def predictHousePrice(trData, distValues, k):
	maxValue = 0
	IndexArr = []
	sortedDist = []
	tmp = 0.0
	currTmp = 0.0
	indxTmp = 0
	Sum = 0.0
	
	for i in distValues:
		if i > maxValue:
			maxValue = i
	maxValue = maxValue + 1
	currTmp = maxValue
	
	for m in range(k):
		for n in range(len(distValues)):
			tmp = distValues[n]
			if tmp < currTmp and tmp not in sortedDist:
				currTmp = tmp
				indxTmp = n
		sortedDist.append(currTmp)
		IndexArr.append(indxTmp)
		currTmp = maxValue
	
	for i in IndexArr:
		Sum = Sum + trData[i]
	Sum = Sum / len(IndexArr)
	return Sum

def main():
	#TASK 1
	X1 = [2.5, 3.6, 1.2, 0.8, 4.0, 3.4]
	X2 = [1.2, 1.0, 1.8, 0.9, 3.0, 2.2]
	X3 = [8.0, 15.0, 12.0, 6.0, 8.0, 10.0]
	
	#Find the correlations between the variable vectors: X1 & X2, X1 & X3 and X2 & X3 using correlation coeffcient approach.
	print("\nTask 1, the following are the correlation coefficient in the order X1X2,X1X3 and X2X3: \n")
	X1X2 = correlationCoeff(X1,X2)
	X1X3 = correlationCoeff(X1,X3)
	X2X3 = correlationCoeff(X2,X3)
	print("X1X2",X1X2)
	print("X1X3",X1X3)
	print("X2X3",X2X3)
	
	#TASK 2
	#Training dataset
	testTemp = []
	trTemp = []
	distValues = []
	DisTemp = 0.0
	PriceHouse = [500000, 800000, 1000000, 350000, 100000]
	NumRooms = [2, 3, 6, 2, 2]
	SizeHouse = [45, 65, 100, 30, 25]
	AgeHouse = [25, 30, 40, 20, 20]
	
	#Testing dataset - PriceHouse dataset is unknown
	PriceHouse2 = [0, 0]
	NumRooms2 = [4, 1]
	SizeHouse2 = [100, 60]
	AgeHouse2 = [25, 20]
	
	for j in range(len(NumRooms2)):
		testTemp = []
		testTemp.append(NumRooms2[j])
		testTemp.append(SizeHouse2[j])
		testTemp.append(AgeHouse2[j])
		for i in range(len(NumRooms)):
			trTemp = []
			trTemp.append(NumRooms[i])
			trTemp.append(SizeHouse[i])
			trTemp.append(AgeHouse[i])
			DisTemp = euclideanDistance(testTemp, trTemp)
			distValues.append(DisTemp)
	
	testData1 = distValues[:(len(distValues)/2)]
	testData2 = distValues[(len(distValues)/2):]
	print("\nTask 2: \n")
	predictHouse = predictHousePrice(PriceHouse, testData1, 1)	
	print("(K = 1)Price of House for the first index data in testing data is predicted to be: ")
	print(predictHouse)
	predictHouse = predictHousePrice(PriceHouse, testData2, 1)		
	print("(K = 1)Price of House for the second index data in testing data is predicted to be: ")
	print(predictHouse)
	predictHouse = predictHousePrice(PriceHouse, testData1, 2)	
	print("\n(K = 2)Price of House for the first index data in testing data is predicted to be: ")
	print(predictHouse)
	predictHouse = predictHousePrice(PriceHouse, testData2, 2)		
	print("(K = 2)Price of House for the second index data in testing data is predicted to be: ")
	print(predictHouse)
	print("\n")
	
main()
