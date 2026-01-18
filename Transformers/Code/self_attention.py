import numpy as np

# Set seed to obtain the same random numbers
np.random.seed(3)

# Number of inputs
N = 3
# Number of dimensions of each input
D = 4

# Create an empty list for the input value
all_x = []

# Create elements x_n and append to list of input values
for n in range(N):
  all_x.append(np.random.normal(size=(D,1)))

# Print out the input values
print(all_x)

# Set seed so we get the same random numbers
np.random.seed(0)

# Choose random values for the parameters
omega_q = np.random.normal(size=(D,D))
omega_k = np.random.normal(size=(D,D))
omega_v = np.random.normal(size=(D,D))
beta_q = np.random.normal(size=(D,1))
beta_k = np.random.normal(size=(D,1))
beta_v = np.random.normal(size=(D,1))

# Define three lists to store queries, keys, and values
all_queries = []
all_keys = []
all_values = []

# For every input value (x)
for x in all_x:
  # Compute query, key, and value
  query = beta_q + omega_q @ x
  key = beta_k + omega_k @ x
  value = beta_v + omega_v @ x

  # Append value of query, key, and value to their respective lists
  all_queries.append(query)
  all_keys.append(key)
  all_values.append(value)

# Define the softmax function
def softmax(items_in):
  scores = np.array(items_in, dtype=float).flatten()
  scores = scores - np.max(scores)
  exp_scores = np.exp(scores)
  items_out = exp_scores / np.sum(exp_scores)
  return items_out

# Define an empty list for the output
all_x_prime = []
all_values = np.array(all_values)

# For each output
for n in range(N):
  # Create list for dot products of query N with all keys
  all_km_qn = []

  # For each key value in all key values
  for key in all_keys:
    # Compute dot product between query and key
    dot_product = float(all_queries[n].T @ key)
    # Store output of dot product
    all_km_qn.append(dot_product)

  # Compute attention weights (1D vector length N)
  attention = softmax(all_km_qn)
  # Print the resulting attention weights
  print("Attentions for output ", n)
  print(attention)
  # Define array structure to store output value
  x_prime = np.zeros((D,))
  
  # Compute weighted sum of values using its corresponding attention weights
  for i in range(N):
    x_prime += all_values[i].flatten() * attention[i]
  all_x_prime.append(x_prime)

# Compare calculated values to true values to ensure correctness
print("x_prime_0_calculated:", all_x_prime[0].transpose())
print("x_prime_0_true: [[ 0.94744244 -0.24348429 -0.91310441 -0.44522983]]")
print("x_prime_1_calculated:", all_x_prime[1].transpose())
print("x_prime_1_true: [[ 1.64201168 -0.08470004  4.02764044  2.18690791]]")
print("x_prime_2_calculated:", all_x_prime[2].transpose())
print("x_prime_2_true: [[ 1.61949281 -0.06641533  3.96863308  2.15858316]]")
