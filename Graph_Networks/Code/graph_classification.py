import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define a graph that represents the chemical structure of ethanol where:
# Each node is labelled with the node number and the element (carbon, hydrogen, oxygen)
G = nx.Graph()
G.add_edge('0:H','2:C')
G.add_edge('1:H','2:C')
G.add_edge('3:H','2:C')
G.add_edge('2:C','5:C')
G.add_edge('4:H','5:C')
G.add_edge('6:H','5:C')
G.add_edge('7:O','5:C')
G.add_edge('8:H','7:O')
nx.draw(G, nx.spring_layout(G, seed = 0), with_labels=True, node_size=600)

# Draw the defined graph
plt.show()

# Define adjacency matrix for ethanol
# Define nodes
nodes = G.nodes()

# Define edges
edges = G.edges()

# Define array structure for adjacency matrix
A = np.zeros((9,9))

# Loop over all the edges and put 1 between nodes where an edge exists
for u, v in G.edges():
    i, j = int(u.split(':')[0]), int(v.split(':')[0])  # Convert labels to numerical indices
    A[i, j] = 1
    A[j, i] = 1

# Print the adjacency matrix for ethanol
print(A)


# Define node matrix for ethanol, where each column represents a node and its element
# Define list structure for separated nodes
separated_nodes = []

# Define array structure for node matrix
X = np.zeros((118,9))

# Split the nodes into just its element
for u in G.nodes():
    i = (u.split(':')[1]) # Convert labels to element indices
    separated_nodes.append(i) 

# Loop over the separated nodes and update the node matrix based on which element each node represents
for i in range(len(separated_nodes)):  
    if separated_nodes[i] == 'H': 
        X[0, i] = 1
    elif separated_nodes[i] == 'C':
        X[5, i] = 1
    elif separated_nodes[i] == 'O':
        X[7, i] = 1

# Print the top 15 rows of the node matrix
print(X[0:15,:])

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

# Define the logistic sigmoid function
def sigmoid(x):
  return 1.0/(1.0+np.exp(-x))

# Define hyperparameters for GCN, where K is the number of hidden layers and D is its dimensions
K = 3; D = 200

# Set seed to always obtain the same random numbers
np.random.seed(1)

# Initialize the parameter matrices randomly with He initialization
Omega0 = np.random.normal(size=(D, 118)) * 2.0 / D
beta0 = np.random.normal(size=(D,1)) * 2.0 / D
Omega1 = np.random.normal(size=(D, D)) * 2.0 / D
beta1 = np.random.normal(size=(D,1)) * 2.0 / D
Omega2 = np.random.normal(size=(D, D)) * 2.0 / D
beta2 = np.random.normal(size=(D,1)) * 2.0 / D
omega3 = np.random.normal(size=(1, D))
beta3 = np.random.normal(size=(1,1))

# Define graph convolutional neural network (GCN)
def graph_neural_network(A, X, Omega0, beta0, Omega1, beta1, Omega2, beta2, omega3, beta3):
  A_I = A + np.eye(len(A), dtype=float)
  hidden_layer_1 = ReLU(beta0 + Omega0 @ (X @ A_I))
  hidden_layer_2 = ReLU(beta1 + Omega1 @ (hidden_layer_1 @ A_I))
  hidden_layer_3 = ReLU(beta2 + Omega2 @ (hidden_layer_2 @ A_I))
  hidden_layer_3 = hidden_layer_3.mean(axis=1, keepdims=True)
  f = sigmoid(beta3 + (omega3 @ hidden_layer_3))

  return f

# Execute the GCN with defined parameters and print results
f = graph_neural_network(A,X, Omega0, beta0, Omega1, beta1, Omega2, beta2, omega3, beta3)
print("Computed value is %3.3f: "%(f[0,0]), "True value of f: 0.310843")

# Define a permutation matrix
P = np.array([[0,1,0,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,0,1],
              [1,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,1,0,0]])

# Permute the adjacency matrix A and node matrix X
A_permuted = P @ A
X_permuted = X @ P

# Execute the GCN with the premutated adjacency matrix and node matrix
f = graph_neural_network(A_permuted,X_permuted, Omega0, beta0, Omega1, beta1, Omega2, beta2, omega3, beta3)
print("Your value is %3.3f: "%(f[0,0]), "True value of f: 0.310843")
     