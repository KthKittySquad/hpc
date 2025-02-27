import numpy as np

def g(x):
	""" sigmoid function """
	return 1.0 / (1.0 + np.exp(-x))


def grad_g(x):
	""" gradient of sigmoid function """
	gx = g(x)
	return gx * (1.0 - gx)	


def predict(Theta1, Theta2, X):
	""" Predict labels in a trained three layer classification network.
	Input:
	  Theta1       trained weights applied to 1st layer (hidden_layer_size x input_layer_size+1)
	  Theta2       trained weights applied to 2nd layer (num_labels x hidden_layer_size+1)
	  X            matrix of training data      (m x input_layer_size)
	Output:     
	  prediction   label prediction
	"""
	
	m = np.shape(X)[0]                    # number of training values
	num_labels = np.shape(Theta2)[0]
	
	a1 = np.hstack((np.ones((m,1)), X))   # add bias (input layer)
	a2 = g(a1 @ Theta1.T)                 # apply sigmoid: input layer --> hidden layer
	a2 = np.hstack((np.ones((m,1)), a2))  # add bias (hidden layer)
	a3 = g(a2 @ Theta2.T)                 # apply sigmoid: hidden layer --> output layer
	
	prediction = np.argmax(a3,1).reshape((m,1))
	return prediction


def reshape(theta, input_layer_size, hidden_layer_size, num_labels):
	""" reshape theta into Theta1 and Theta2, the weights of our neural network """
	ncut = hidden_layer_size * (input_layer_size+1)
	Theta1 = theta[0:ncut].reshape(hidden_layer_size, input_layer_size+1)
	Theta2 = theta[ncut:].reshape(num_labels, hidden_layer_size+1)
	return Theta1, Theta2


def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
	""" Neural net cost function gradient for a three layer classification network.
	Input:
	  theta               flattened vector of neural net model parameters
	  input_layer_size    size of input layer
	  hidden_layer_size   size of hidden layer
	  num_labels          number of labels
	  X                   matrix of training data
	  y                   vector of training labels
	  lmbda               regularization term
	Output:
	  grad                flattened vector of derivatives of the neural network 
	"""
	
	# unflatten theta
	Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
	
	# number of training values
	m = len(y)
	
	# Backpropagation: calculate the gradients Theta1_grad and Theta2_grad:
	
	Delta1 = np.zeros((hidden_layer_size,input_layer_size+1))
	Delta2 = np.zeros((num_labels,hidden_layer_size+1))

	for t in range(m):
		
		# forward very bad, very slow
		a1 = X[t,:].reshape((input_layer_size,1))
		a1 = np.vstack((1, a1))   #  +bias 		 				<--------- 12.7%
		z2 = Theta1 @ a1 # 										<--------- 7.5%
		a2 = g(z2)	#                     		       	    	<--------- 5.9%
		a2 = np.vstack((1, a2))   #  +bias				    	<--------- 9.7%
		a3 = g(Theta2 @ a2) # 									<--------- 7.7%
		
		# compute error for layer 3
		y_k = np.zeros((num_labels,1))
		y_k[y[t,0].astype(int)] = 1
		delta3 = a3 - y_k
		Delta2 += (delta3 @ a2.T)
		
		# compute error for layer 2
		delta2 = (Theta2[:,1:].T @ delta3) * grad_g(z2)  # 		<--------- 11.9%
		Delta1 += (delta2 @ a1.T)	 #                     		<--------- 33.4%

	Theta1_grad = Delta1 / m
	Theta2_grad = Delta2 / m

	# add regularization
	Theta1_grad[:,1:] += (lmbda/m) * Theta1[:,1:]	
	Theta2_grad[:,1:] += (lmbda/m) * Theta2[:,1:]

	# flatten gradients
	grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

	return grad
