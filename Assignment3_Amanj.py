import sys
import math as m
from numpy import *
import numpy as np
from sklearn.cluster import KMeans

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


######TASK3 - OPTIONAL#####
	return None






main()
