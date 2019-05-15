import sys
import math as m
from numpy import *
import numpy as np

def softmax(x):
	"Source: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

def main():
#######TASK1###############
	print("\nTask1: Questions\n")
	print("a) How many weights and biases does the RNN model have?\n10 weights and 4 biases\n")
	
	print("b) How many input-to-hidden weights does the RNN have?\n6 weights\n")

	print("c) How many hidden node biases does the RNN have?\n2 biases\n")

	print("d) How many hidden-to-output weights does the RNN have?\n4 weights\n")

	print("e) How many output nodes does the RNN have?\n2 Output nodes\n")

	print("f) What are the input values in the RNN?\n1.0, 2.0 and 3.0\n")

	print("g) Apply 1 time iteration to the network model to find the values of hidden nodes and output nodes. (USE tanh as activation function and Softmax Function).\n")

	print("h) Apply Elman approach to store the hidden values into the state nodes. Show the figure and explain.\n")
	print("\n3 input nodes In1, In2, In3\n2 hidden nodes H1 and H2\n2 ouput nodes Out1 and Out2\n")
	print("In1 has value 1.0 and, weight 0.01 -> H1, weight 0.01 -> H2")
	print("In2 has value 2.0 and, weight 0.02 -> H1, weight 0.04 -> H2")
	print("In3 has value 3.0 and, weight 0.03 -> H1, weight 0.03 -> H2")
	print("Hidden and output node  biases: H1 bias = 0.17, H2 bias = 0.12, Out1 bias = 0.03 and Out2 bias = 0.08\n")
	print("Calculations for hidden nodes:")
	print("H1: (1.0)(0.01) + (2.0)(0.02) + (3.0)(0.03) + 0.17 = 0.31 -> tanh(0.31) = 0.3004370971476541256605222527790252436631646497813197")
	print("H2: (1.0)(0.01) + (2.0)(0.04) + (3.0)(0.03) + 0.12 = 0.3 -> tanh(0.3) = 0.2913126124515909058182212728237659281535968049176121\n")
	print("\nCalculations for output nodes with tanh activation fucntion:")
	print("Out1: (0.3004370971476541256605222527790252436631646497813197)(0.13) + (0.2913126124515909058182212728237659281535968049176121)(0.45) + 0.03\n = 0.200147498232410943954067465631967949345329966684497006 -> tanh(0.200147498232410943954067465631967949345329966684497006) = 0.197517068238493915817328111541186358962360872184574176")
	print("Out1: (0.3004370971476541256605222527790252436631646497813197)(0.25) + (0.2913126124515909058182212728237659281535968049176121)(0.13) + 0.08\n = 0.192979913905620349171499328661845881575758747084619498 -> tanh(0.192979913905620349171499328661845881575758747084619498) = 0.190619465867508038271026414263182163918924218360534969\n")
	print("Calculations for Softmax Function on the output nodes:")
	softmax_arr = softmax([0.197517068238493915817328111541186358962360872184574176, 0.190619465867508038271026414263182163918924218360534969])
	print("Out1 =",softmax_arr[0])
	print("Out2 =",softmax_arr[1])
	
#######TASK2################
	print("\nTask2: Questions\n")
	print("a) Let’s assume that the input image size is 416x416x1 and the following information for CNN is given\n")
	
	print("CL1: 5x5 kernel size, 32 kernels, padding 1, Stride=1, RELU poollayer = [2 2]\n")

	print("W=416, H=416, K=5, S=1, P=1, 32 filters")
	print("O = ((W-K + 2P)/S) + 1 = ((416-5 + 2)/1) + 1 = 414")
	print("We applied 32 filters then we will have 414 x 414 x 32 feature sets")
	print("After applying Relu and pool layer – it will be 207 x 207 x 32\n")

	print("b) Let’s assume that the input image size is 128x128x1 and the following information for CNN is given\n")
	
	print("CL1: 3x3 kernel size, 32 kernels,   padding 1, Stride=1, RELU poollayer = [2 2]")
	print("CL2: 7x7 kernel size, 16 kernels,   padding 1, Stride=1, RELU poollayer = [2 2]")
	print("CL3: 5x5 kernel size, 128 kernels,  padding 1, Stride=1, RELU poollayer = [2 2]")

	print("CL1")
	print("W=128, H=128, K=3, S=1, P=1, 32 filters")
	print("O = ((W-K + 2P)/S) + 1 = ((128-3 + 2)/1) + 1 = 128")
	print("32 filters --> 128 x 128 x 32 feature sets")
	print("After applying Relu and pool layer –> 64 x 64 x 32\n")
	
	print("CL2")
	print("W=64, H=64, K=7, S=1, P=1, 16 filters")
	print("O = ((W-K + 2P)/S) + 1 = ((64-7 + 2)/1) + 1 = 60")
	print("16 filters --> 60 x 60 x 16 feature sets")
	print("After applying Relu and pool layer –> 30 x 30 x 32\n")
	
	print("CL3")
	print("W=30, H=30, K=5, S=1, P=1, 128 filters")
	print("O = ((W-K + 2P)/S) + 1 = ((30-5 + 2)/1) + 1 = 28")
	print("128 filters --> 28 x 28 x 128 feature sets")
	print("After applying Relu and pool layer –> 14 x 14 x 128\n")

######TASK3 - OPTIONAL#####
	return None






main()
