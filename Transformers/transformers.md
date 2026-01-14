# Chapter 12: Transformers

## Dot-Product Self-Attention 
**<ins>Dot-Product Self-Attention</ins>** is a mechanism that allows for the computation of attention weights, which is crucial in transformers 

### Attention Weights 
<ins>Methods</ins>: To compute the attention weights, we first:

1. Apply two different linear transformation to the input vector, where one linear transformation outputs queries and the other linear transformation outputs keys 

2. Compute the dot product between queries and keys 

3. Pass the output of the dot product through a softmax function  

Using the computed attention weights, we are now able to undergo self-attention

### Self-Attention Methods 
1. For each input vector, put the input vector through a linear transformation to output an output vector

2. Multiply the computed output vector by the attention weights that controls the magnitude/significance of input vectors on the output

3. Repeat this process for all the other input vectors 

<ins>Results</ins>: The use of self-attention allows for shared parameters and strength-varying connections between inputs

### Extensions of Dot-Product Self-Attention 
#### Positional Encoding 
**<ins>Positional encoding</ins>** incorporates the order of the input into the self-attention mechanism. There are two types of positional encoding 

- <ins>Absolute Positional Encoding</ins>: A matrix with positional information is added to the input 

- <ins>Relative Positional Encoding</ins>: Relative positional encodings are added to the attention weights, which influences output values based on relative position of input values 

#### Scaled Dot-Product Self-Attention 
**<ins>Scaled dot-product self-attention</ins>** scales the value of the attention weights so that bigger inputs don’t overshadow smaller inputs on the overall self-attention method 

#### Multiple Heads 
**<ins>Multi-head self attention</ins>** is a mechanism where multiple attention weights, each with a different set of queries and keys, are computed. The output vector of each self-attention method are concatenated to form a final output vector 

## Transformer Layer 
After the dot-product self-attention is computed, the output is…

1. Inputted into a LayerNorm operation, to stabilize the output.  

2. The output of the LayerNorm operation is run through a fully connected network to introduce nonlinearity (essentially just the same as a neural network model)

3. The output of the fully connected network is run through another LayerNorm operation to stabilize the output 

## Application of Transformers for Natural Language Processing 
### Tokenizer
**<ins>Tokenizer</ins>**: a tokenizer splits a text into smaller units (tokens) 

- Often, the tokens produced by a tokenizer often consists of complete words or word fragments (sub-word tokenizer) 

### Embedding 
After a tokenizer splits the text into tokens, it displays each token as a vector representation of high dimensions through embeddings 

### Transformer Model
The data from embeddings is run through a series of transformer layers to produce a transformer model 

<ins>Types of Transformer Models</ins>: 

- Encoder: uses given text to perform a variety of tasks 

- Decoder: predicts next token/phrase given a text 

- Encoder-Decoder: uses the given text and converts it into another (example: machine translation)