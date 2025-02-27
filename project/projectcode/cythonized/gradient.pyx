import numpy as snp
cimport numpy as np

from helper import g, grad_g, reshape

def gradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
	""" 
		Neural net cost function gradient for a three layer classification network.
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
	cdef np.ndarray[np.float64_t, ndim=2] Theta1, Theta2
	Theta1, Theta2 = reshape(theta, input_layer_size, hidden_layer_size, num_labels)
	
	# number of training values
	cdef int m = len(y)
	
	# Backpropagation: calculate the gradients Theta1_grad and Theta2_grad:
	
	cdef np.ndarray[np.float64_t, ndim=2] Delta1 = snp.zeros((hidden_layer_size,input_layer_size+1))
	cdef np.ndarray[np.float64_t, ndim=2] Delta2 = snp.zeros((num_labels,hidden_layer_size+1))

	cdef int t

	cdef np.ndarray[np.float64_t, ndim=2] a1
	cdef np.ndarray[np.float64_t, ndim=2] z2
	cdef np.ndarray[np.float64_t, ndim=2] a2 
	cdef np.ndarray[np.float64_t, ndim=2] a3

	cdef np.ndarray[np.float64_t, ndim=2] y_k
	cdef np.ndarray[np.float64_t, ndim=2] delta3 

	cdef np.ndarray[np.float64_t, ndim=2] delta2
	for t in range(m):
		
		# forward very bad, very slow
		a1 = X[t,:].reshape((input_layer_size,1))
		a1 = snp.vstack((1, a1))   #  +bias 		 				<--------- 12.7%
		z2 = Theta1 @ a1 # 										<--------- 7.5%
		a2 = g(z2)	#                     		       	    	<--------- 5.9%
		a2 = snp.vstack((1, a2))   #  +bias				    	<--------- 9.7%
		a3 = g(Theta2 @ a2) # 									<--------- 7.7%
		
		# compute error for layer 3
		y_k = snp.zeros((num_labels,1))
		y_k[y[t,0].astype(int)] = 1
		delta3 = a3 - y_k
		Delta2 += (delta3 @ a2.T)
		
		# compute error for layer 2
		delta2 = (Theta2[:,1:].T @ delta3) * grad_g(z2)  # 		<--------- 11.9%
		Delta1 += (delta2 @ a1.T)	 #                     		<--------- 33.4%

	cdef np.ndarray[np.float64_t, ndim=2] Theta1_grad = Delta1 / m
	cdef np.ndarray[np.float64_t, ndim=2] Theta2_grad = Delta2 / m

	# add regularization
	Theta1_grad[:,1:] += (lmbda/m) * Theta1[:,1:]	
	Theta2_grad[:,1:] += (lmbda/m) * Theta2[:,1:]

	# flatten gradients 
	cdef np.ndarray[np.float64_t, ndim=1] grad = snp.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))

	return grad
