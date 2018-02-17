
import numpy as np
import matplotlib.pyplot as plt

#################
# DATA PROCESSING FUNCTIONS

# Normalize each column of the matrix to be in the range [low, high]
def scale_linear_by_column(rawpoints, high=1.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = (maxs - mins)*0.5
    return high - (((high - low) * (maxs - rawpoints)) / rng)

# Normalizes and shift the data
def preprocess_data(x):
	#normalize the data
	x=scale_linear_by_column(x) # np.linalg.norm(x, axis=0)
	# shift the data
	x=x-np.mean(x, axis=0)	
	return x

#################
# KERNEL FUNCTIONS

# Gaussian
def probability(x, u, sigma2):
	sigma=np.sqrt(sigma2)
	p=np.divide(1, np.sqrt(2*np.pi)*sigma, out=np.zeros_like(sigma), where=sigma!=0) 
	p=p*np.exp(-np.divide(np.square(x-u), 2*sigma2))
	return np.prod(p, axis=1)

# Multivariate Gaussian
def probability_multivariate(x, u, E):
	n=x[0,:].size
	p1=np.dot(np.linalg.pinv(E),(x-u).T)
	p2=np.exp(-0.5*((x-u)* p1.T))
	p=np.divide(p2, np.power(np.power(2*np.pi, n/2)*np.linalg.det(E),0.5)) 
	return np.prod(p, axis=1)

###################
# MAIN ALGORITHM

def main(unused_argv):  
	# input to the algorithm                                                         
	m=6 			# number of examples
	m_test=3000 	# number of tests
	n=2				# number of features for each example. Here is 2 so we can plot directly.
	x_orig=np.random.normal(size=[m,n]) 	# generate training data
	x_test_orig=np.random.rand(m_test,n)	# generate test data

	x=preprocess_data(x_orig)					# mean shift normalize
	x_test=preprocess_data(x_test_orig)*2		# mean shift normalize

	u=np.sum(x,axis=0)/m 					# mean
	sigma2=np.sum(np.square(x-u),axis=0)/m 	# variance
	p=probability(x_test, u, sigma2)		# the probability that an example is an anomaly or not

	E=np.dot((x-u).T,(x-u))/m 				# covariance matrix
	#E=np.cov(x.T)				 			# covariance matrix
	p_multi=probability_multivariate(x_test, u, E)		# the probability that an example is an anomaly or not

	epsilon=0.182 	# hand-picked threshold for flagging an example as an anomaly



	# Plotting
	plt.figure(1)

	# plot predictions using gaussian distribution
	plt.subplot(211)
	plt.plot(x_test[p>=epsilon,0], x_test[p>=epsilon,1], "y+")	# 
	plt.plot(x_test[p<epsilon,0], x_test[p<epsilon,1], "rx")	# anomalous points
	plt.plot(x[:,0], x[:,1], "bo")	# data points

	# plot predictions using multivariate gaussian distribution
	plt.subplot(212)
	plt.plot(x_test[p_multi>=epsilon,0], x_test[p_multi>=epsilon,1], "y+")
	plt.plot(x_test[p_multi<epsilon,0], x_test[p_multi<epsilon,1], "rx")
	plt.plot(x[:,0], x[:,1], "bo")

	plt.show()



if __name__ == "__main__":
	main(main)