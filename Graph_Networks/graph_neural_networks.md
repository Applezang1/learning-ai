# Chapter 13: Graph Neural Networks 
## Graphs 
**<ins>Graph</ins>***: a structure that consists of nodes/vertices, where different nodes are connected by edges/links 

<ins>Example of Graphs</ins>: 

- <ins>Electrical Circuits</ins>, where the nodes are the components and the edges are the electrical connections between components

- <ins>Chemical Molecules</ins>, where the nodes are the atoms and the edges are the chemical bonds 

### Types of Graphs 
- <ins>Undirected Edges</ins>: Connections between different nodes are not directional and the connections therefore are both-way 

- <ins>Directed Edges</ins>: Connections between different nodes are directional and the connections are inherently one-way 

- <ins>Knowledge Graph</ins>: Graph of an object along with a set of facts about it, where connections define the relationship between the different facts 

    - Also known as a directed heterogenous multigraph, where heterogenous indicates that nodes can represent different entities and multigraph means that there are multiple types of edges between nodes 

- <ins>Geometric Graph</ins>: Graph where a node represents a position in 3D space 

- <ins>Hierarchical Graph</ins>: Graph where nodes and edges describe the adjacency between components (house furniture, for example) 

Information is often stored in the nodes and the edges. These information are called <ins>node embedding</ins> and </ins>edge embedding</ins> respectively 

### Graph Encoding Matrices 
In addition, the information of the graph (such as graph structure, node embedding) can be represented by matrices, where the matrices are defined as: 

#### Adjacency Matrix (A) 
<ins>Structure</ins>: N X N (node X node) 

<ins>Representation</ins>: The adjacency matrix represents (m, n), where m and n are two different nodes. If there is an edge between nodes m and n, the matrix value for that entry is set to 1 and 0 if there is no edge 

#### Node Data Matrix (X)
<ins>Structure</ins>: D X N (length of node X node) 

<ins>Representation</ins>: The node data matrix represents the concatenated node embeddings 

#### Edge Data Matrix (E)
<ins>Structure</ins>: D(e) X E (length of edge X edge) 

<ins>Representation</ins>: The edge data matrix represents the edge embeddings 

## Graph Neural Networks 
Graph Neural Networks take node embeddings (X) and adjacency matrix (A) as inputs, which are passed through hidden layers, resulting in output embeddings. 

### Usage of Graph Neural Networks: 
**<ins>Graph-level tasks</ins>**: Classifying the type of graph 

<ins>Procedure</ins>: 

- Graph is passed through a GNN (graph neural network) 

- The output of the GNN (node embeddings) are combined using mean pooling 

- The combined node embeddings are mapped through a linear transformation/neural network to output a distribution of possible class probabilities 

**<ins>Node-level tasks</ins>**: Classifying each node of the graph, similar procedure to graph-level tasks except for the fact that it’s done for each node of the graph 

**<ins>Edge-prediction tasks</ins>**: Predicts whether an edge should exist between two different nodes or not. 

### Graph Convolutional Networks (GCNs) 
**<ins>Graph Convolutional Networks</ins>**: GCNs update each node embedding by getting information from neighboring nodes. This is defined as a <ins>relational inductive bias</ins>, which prioritizes information from neighbors. 

- **<ins>Invariance</ins>**: Invariance is a property that states that the output is not affected by the permutation of node embeddings. Invariance applies to graph-level tasks, where the final graph classification output is invariant to the node order

- **<ins>Equivariance</ins>**: Equivariance is a property that states that the output is affected by the permutation of node embeddings. Equivariance applies to node-level tasks, where the classification of a node is equivariant to the node order 

- **<ins>Parameter Sharing</ins>**: GCNs updates parameters of the nodes by aggregating the sum of information from neighboring nodes. Unlike transformers however, it’s impossible to assign different weights to different nodes when aggregating information 

#### Graph Convolutional Networks Procedure: 

- Aggregate information from neighboring nodes by summing their node embeddings 

- Take the weighted sum of the aggregated information 

- Incorporate the weighted sum into a linear transformation of the current embedding 

- Pass the linear transformation through a nonlinear activation function to compute the hidden unit  

## Transductive Models: 
A **<ins>transductive model</ins>** is a type of model that uses both labeled and unlabeled data and labels the unlabeled data based on trends of the overall graph. This is sometimes also known as semi-supervised learning. 

### Application of Transductive Model 
#### Node Classification: 
In node classification, we use transductive models to label the remaining unlabeled nodes. This is done by using a GCN to output predictions of labels for each of the remaining unlabeled nodes.  

<ins>Loss Function</ins>: The loss of the GCN can be computed by using the binary cross-entropy loss only for nodes which are labeled 

<ins>Batches</ins>: 
In order to form batches for stochastic gradient descent of the GCN, we can choose a random subset of labeled nodes.

However, this might not always work because if the label of each node depends on the other, not including some of the nodes complicates training. This is known as the <ins>graph expansion problem</ins>

<ins>Solutions</ins>: 

- <ins>Neighborhood Sampling</ins>: Neighborhood sampling turns the graph into sub-graphs, where each subgraph has a fixed number of neighbors. These subgraphs are inputted into the stochastic gradient descent step for training. 

- <ins>Graph Partitioning</ins>: Graph partitioning separates the original graph into a subset of nodes, where each subset of nodes have internal links to other nodes in the subset.  

## Modified Methods of Computing GCN Layer 
The <ins>Graph Convolutional Networks Procedure</ins> (defined above) can be modified in different ways. The following methods are: 

- **<ins>Diagonal Enhancement</ins>**: Multiplying the current node/embedding by (1 + e), where e is a learned parameter that is different for different layers. 

    - <ins>Variation</ins>: Applying a linear transformation to the current node before passing the current embedding and the aggregated neighbors through a nonlinear activation function 

- Incorporating residual connections to the GCN layer 

- **<ins>Mean Aggregation</ins>**: Instead of taking the weighted sum of the aggregated information from neighbors, take the average of the neighbors.  

- **<ins>Kipf Normalization</ins>**: Kipf normalization is a type of mean aggregation where the current node is also included in the mean computation 

- **<ins>Max Pooling Aggregation</ins>**: Taking the maximum of the neighbors and using that as the aggregated information 

- **<ins>Graph Attention</ins>**: Instead of weighting the neighboring information equally, attention methods can be used to scale weight values based on the neighbor 

## Edge Graph 
Edge graphs take the original graph and make the edges the new nodes and the nodes the new edges. This allows edge embeddings to be computed exactly the same as node embeddings. 
