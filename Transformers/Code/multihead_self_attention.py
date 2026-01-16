import numpy as np
import matplotlib.pyplot as plt

# Set seed so we get the same random numbers
np.random.seed(3)

# Define the number of inputs
N = 6

# Define the number of dimensions of each input
D = 8

# Initalize an empty list as input data
X = np.random.normal(size=(D,N))

# Define the number of heads for the multi-head self-attention
H = 2
 
# Define the dimensions of QKV (value, query, key)
H_D = int(D/H)

# Set seed so we get the same random numbers
np.random.seed(0)

# Initialize random values for the parameters for the first head
omega_q1 = np.random.normal(size=(H_D,D))
omega_k1 = np.random.normal(size=(H_D,D))
omega_v1 = np.random.normal(size=(H_D,D))
beta_q1 = np.random.normal(size=(H_D,1))
beta_k1 = np.random.normal(size=(H_D,1))
beta_v1 = np.random.normal(size=(H_D,1))

# Initialize random values for the parameters for the second head
omega_q2 = np.random.normal(size=(H_D,D))
omega_k2 = np.random.normal(size=(H_D,D))
omega_v2 = np.random.normal(size=(H_D,D))
beta_q2 = np.random.normal(size=(H_D,1))
beta_k2 = np.random.normal(size=(H_D,1))
beta_v2 = np.random.normal(size=(H_D,1))

# Initialize random values for the parameters of the model
omega_c = np.random.normal(size=(D,D))

# Define the softmax function
def softmax_cols(data_in):
  # Exponentiate all of the values (numerator)
  exp_values = np.exp(data_in) 
  # Sum the exponentiated values over all columns (denominator)
  denom = np.sum(exp_values, axis = 0)
  # Compute softmax function
  softmax = exp_values / denom

  return softmax

# Define a function to compute multihead scaled self attention in matrix form
def multihead_scaled_self_attention(X,omega_v1, omega_q1, omega_k1, beta_v1, beta_q1, beta_k1, omega_v2, omega_q2, omega_k2, beta_v2, beta_q2, beta_k2, omega_c):
  # Define array structure for the output data
  X_prime = np.zeros_like(X) 

  # Compute QKV values for the first heaad
  all_values_1 = beta_v1 + (omega_v1 @ X)
  all_query_1 = beta_q1 + (omega_q1 @ X)
  all_key_1 = beta_k1 + (omega_k1 @ X)

  # Compute QKV values for the second head
  all_values_2 = beta_v2 + (omega_v2 @ X) 
  all_query_2 = beta_q2 + (omega_q2 @ X) 
  all_key_2 = beta_k2 + (omega_k2 @ X) 

  # Compute the self attention weights for the first and second head 
  self_attention_1 = all_values_1 @ softmax_cols(all_key_1.T @ all_query_1/np.sqrt(all_key_1.shape[0]))
  self_attention_2 = all_values_2 @ softmax_cols(all_key_2.T @ all_query_2/np.sqrt(all_key_2.shape[0]))

  # Concatenate the 2 computed self attention weights into one array
  all_self_attention = np.concatenate([self_attention_1.T, self_attention_2.T], axis = 1)

  # Compute dot product with concatenated array and weight to compute the output
  X_prime = omega_c @ all_self_attention.T

  return X_prime

# Undergo the self attention mechanism
X_prime = multihead_scaled_self_attention(X,omega_v1, omega_q1, omega_k1, beta_v1, beta_q1, beta_k1, omega_v2, omega_q2, omega_k2, beta_v2, beta_q2, beta_k2, omega_c)

# Print the outputs of the self attention mechanism
np.set_printoptions(precision=3)
print("Your answer:")
print(X_prime)

print("True values:")
print("[[-21.207  -5.373 -20.933  -9.179 -11.319 -17.812]")
print(" [ -1.995   7.906 -10.516   3.452   9.863  -7.24 ]")
print(" [  5.479   1.115   9.244   0.453   5.656   7.089]")
print(" [ -7.413  -7.416   0.363  -5.573  -6.736  -0.848]")
print(" [-11.261  -9.937  -4.848  -8.915 -13.378  -5.761]")
print(" [  3.548  10.036  -2.244   1.604  12.113  -2.557]")
print(" [  4.888  -5.814   2.407   3.228  -4.232   3.71 ]")
print(" [  1.248  18.894  -6.409   3.224  19.717  -5.629]]")