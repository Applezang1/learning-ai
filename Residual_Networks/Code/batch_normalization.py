import numpy as np
import os
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import mnist1d
import random

# Import input data and store in variable
args = mnist1d.data.get_dataset_args()
data = mnist1d.data.get_dataset(args, path='./mnist1d_data.pkl', download=False, regenerate=False)

# The training and test input and outputs are in
# data['x'], data['y'], data['x_test'], and data['y_test']
print("Examples in training set: {}".format(len(data['y'])))
print("Examples in test set: {}".format(len(data['y_test'])))
print("Length of each example: {}".format(data['x'].shape[-1]))

# Separate the input data into training and validation data and store in respective variables
train_data_x = data['x'].transpose()
train_data_y = data['y']
val_data_x = data['x_test'].transpose()
val_data_y = data['y_test']

# Print the dimensionsn of the training and validation data
print("Train data: %d examples (columns), each of which has %d dimensions (rows)"%((train_data_x.shape[1],train_data_x.shape[0])))
print("Validation data: %d examples (columns), each of which has %d dimensions (rows)"%((val_data_x.shape[1],val_data_x.shape[0])))

# Define a function to compute the variance between hidden units output per hidden layer 
def print_variance(name, data):
  # First dimension (rows) is batch elements (# of input data)
  # Second dimension (columns) is number of neurons.
  np_data = data.detach().numpy()
  # Compute variance across neurons and average these variances over members of the batch
  neuron_variance = np.mean(np.var(np_data, axis=0))
  # Print out the name and the variance
  print("%s variance=%f"%(name,neuron_variance))

# Define He initialization of weights
def weights_init(layer_in):
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

# Define a backpropagation function
def run_one_step_of_model(model, x_train, y_train):
  # Define cross entropy loss function as the loss function for the residual neural network
  loss_function = nn.CrossEntropyLoss()

  # Define the stochastic gradient descent step and initialize learning rate and momentum
  optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)

  # Load the converted data into a class that creates the batches
  data_loader = DataLoader(TensorDataset(x_train,y_train), batch_size=200, shuffle=True, worker_init_fn=np.random.seed(1))

  # Initialize model weights
  model.apply(weights_init)

  # For each example in the input data
  for i, data in enumerate(data_loader):
    # Retrieve inputs and labels for this batch (example)
    x_batch, y_batch = data
    # Reset the parameter gradients to zero
    optimizer.zero_grad()
    # Compute the forward pass and the model output
    pred = model(x_batch)
    # Compute the loss
    loss = loss_function(pred, y_batch)
    # Compute the backward pass
    loss.backward()
    # Undergo the SGD update
    optimizer.step()

    break

# Convert training and test data into proper format for training
x_train = torch.tensor(train_data_x.transpose().astype('float32'))
y_train = torch.tensor(train_data_y.astype('long'))

# Define a residual neural network model with 5 residual branches
class ResidualNetwork(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(ResidualNetwork, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, hidden_size)
    self.linear4 = nn.Linear(hidden_size, hidden_size)
    self.linear5 = nn.Linear(hidden_size, hidden_size)
    self.linear6 = nn.Linear(hidden_size, hidden_size)
    self.linear7 = nn.Linear(hidden_size, output_size)

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    print_variance("Input",x)
    f = self.linear1(x)
    print_variance("First preactivation",f)
    res1 = f+ self.linear2(f.relu())
    print_variance("After first residual connection",res1)
    res2 = res1 + self.linear3(res1.relu())
    print_variance("After second residual connection",res2)
    res3 = res2 + self.linear4(res2.relu())
    print_variance("After third residual connection",res3)
    res4 = res3 + self.linear5(res3.relu())
    print_variance("After fourth residual connection",res4)
    res5 = res4 + self.linear6(res4.relu())
    print_variance("After fifth residual connection",res5)
    return self.linear7(res5)

# Define hyperparameters for the residual neural network
n_hidden = 100
n_input = 40
n_output = 10

# Definen residual neural network model with defined hyperparameters
model = ResidualNetwork(n_input, n_output, n_hidden)

# Compute the variance between hidden unit outputs for each backward pass
run_one_step_of_model(model, x_train, y_train)

# Define a residual neural network with a batch normalization before each pre-activation
class ResidualNetworkWithBatchNorm(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(ResidualNetworkWithBatchNorm, self).__init__()
    self.batchnorm1 = nn.BatchNorm1d(input_size)
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.batchnorm2 = nn.BatchNorm1d(hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.batchnorm3 = nn.BatchNorm1d(hidden_size)
    self.linear3 = nn.Linear(hidden_size, hidden_size)
    self.batchnorm4 = nn.BatchNorm1d(hidden_size)
    self.linear4 = nn.Linear(hidden_size, hidden_size)
    self.batchnorm5 = nn.BatchNorm1d(hidden_size)
    self.linear5 = nn.Linear(hidden_size, hidden_size)
    self.batchnorm6 = nn.BatchNorm1d(hidden_size)
    self.linear6 = nn.Linear(hidden_size, hidden_size)
    self.batchnorm7 = nn.BatchNorm1d(hidden_size)
    self.linear7 = nn.Linear(hidden_size, output_size)

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    print_variance("Input",x)
    x = self.batchnorm1(x)
    f = self.linear1(x)
    print_variance("First preactivation",f)
    f = self.batchnorm2(f)
    res1 = f+ self.linear2(f.relu())
    print_variance("After first residual connection",res1)
    res1 = self.batchnorm3(res1)
    res2 = res1 + self.linear3(res1.relu())
    print_variance("After second residual connection",res2)
    res2 = self.batchnorm4(res2)
    res3 = res2 + self.linear4(res2.relu())
    print_variance("After third residual connection",res3)
    res3 = self.batchnorm5(res3)
    res4 = res3 + self.linear5(res3.relu())
    print_variance("After fourth residual connection",res4)
    res4 = self.batchnorm6(res4)
    res5 = res4 + self.linear6(res4.relu())
    res5 = self.batchnorm7(res5)
    print_variance("After fifth residual connection",res5)
    return self.linear7(res5)
  
# Define hyperparameters for the residual neural network
n_hidden = 100
n_input = 40
n_output = 10

# Define the residual neural network model with batch normalization
model = ResidualNetworkWithBatchNorm(n_input, n_output, n_hidden)

# Compute the variance between hidden unit outputs for each backward pass
run_one_step_of_model(model, x_train, y_train)