import numpy as np
import matplotlib.pyplot as plt

# Set seed so we get the same random numbers
np.random.seed(1)

# Define the number of nodes in the graph
N = 8
# Define the number of dimensions of each input
D = 4

# Define adjacency graph
A = np.array([[0,1,0,1,0,0,0,0],
              [1,0,1,1,1,0,0,0],
              [0,1,0,0,1,0,0,0],
              [1,1,0,0,1,0,0,0],
              [0,1,1,1,0,1,0,1],
              [0,0,0,0,1,0,1,1],
              [0,0,0,0,0,1,0,0],
              [0,0,0,0,1,1,0,0]])
print(A)

# Define a matrix of random input data
X = np.random.normal(size=(D,N))

# Initialize random values for the parameters
omega = np.random.normal(size=(D,D))
beta = np.random.normal(size=(D,1))
phi = np.random.normal(size=(2*D,1))

# Define a softmax function 
def softmax_cols(data_in):
  # Exponentiate the input value
  exp_values = np.exp(data_in) 
  # Sum over columns
  denom = np.sum(exp_values, axis = 0)
  # Replicate denominator to N rows
  denom = np.matmul(np.ones((data_in.shape[0],1)), denom[np.newaxis,:])
  # Compute softmax
  softmax = exp_values / denom
  return softmax

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

# Define the Graph Attention Network
def graph_attention(X,omega, beta, phi, A):
    # Apply a linear transformation to the input data
    X_prime = beta + omega @ X   

    # Compute all the pairs of the node embeddings
    X_i = np.repeat(X_prime[:, :, np.newaxis], N, axis=2)   
    X_j = np.repeat(X_prime[:, np.newaxis, :], N, axis=1)   
    X_pair = np.concatenate([X_i, X_j], axis=0)             

    # Compute matrix S, which represents the similarity of every node to every other
    S = (phi.T @ X_pair.reshape(2*D, -1)).reshape(N, N)     
    S = ReLU(S) 

    # Mask matrix S to make the attention weights of non-neighboring nodes to zero
    A_hat = A + np.eye(N)
    S_masked = np.where(A_hat==0, -1e20, S)
    softmax_S = softmax_cols(S_masked)

    # Apply the attention weights to the final output
    output = ReLU(X_prime @ softmax_S)
    return output

# Print the output of the Graph Attention Network and Compare with Answers
np.set_printoptions(precision=3)
output = graph_attention(X, omega, beta, phi, A)
print("Correct answer is:")
print("[[0.    0.028 0.37  0.    0.97  0.    0.    0.698]")
print(" [0.    0.    0.    0.    1.184 0.    2.654 0.  ]")
print(" [1.13  0.564 0.    1.298 0.268 0.    0.    0.779]")
print(" [0.825 0.    0.    1.175 0.    0.    0.    0.  ]]]")


print("Your answer is:")
print(output)