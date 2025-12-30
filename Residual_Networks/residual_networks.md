# Chapter 11: Residual Networks 

## Key Words and Concepts: 
**<ins>Shattered Gradients</ins>**: Shattered gradients is the phenomenon of the gradient change for different values of x for shallow and deep neural networks. For shallow neural networks, a change in x results in a relatively slow change in the gradient. For deep neural networks however, a change in x results in a massive and unpredictable change in the gradient. This observations allows us to conclude that nearby gradients often have no correlation with each other, leading to failures to match the training and test data 

## Residual
**<ins>Residuals</ins>**: Residuals are connections that takes the input of the previous hidden layer and adds it to the output of the hidden layer, allowing for better model performance 

- <ins>Note</ins>: With the addition of a residual, this means that the linear transformation must be applied to the input first, followed by the ReLU function and another linear transformation, in order to allow both positive and negative input values to influence the output 

<ins>Problem</ins>: Because the input is added back for each hidden layer, it increases the variance of the neural network model by incorporating the variance of the input. This can lead to too much variance, which leads to the exploding gradient problem 

## Batch Normalization 
Batch normalization fixes this problem by rescaling the activation layers so that the neural network model’s mean and variance are values that it learns during training 

### Benefits of Batch Normalization 
- Better generalization for validation data

- Faster learning rate 

- Stable gradients, avoids the exploding or shrinking gradient problem 
