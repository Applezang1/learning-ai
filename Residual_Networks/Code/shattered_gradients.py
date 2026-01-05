import numpy as np
import matplotlib.pyplot as plt

# Initialize Hyperparameters 
# K is width, D is number of hidden units in each layer
def init_params(K, D):
  np.random.seed(1)

  # Define Input layer
  D_i = 1
  # Define Output layer
  D_o = 1

  # Define Glorot initialization
  sigma_sq_omega = 1.0/D

  # Define list structure to store weights and biases 
  all_weights = [None] * (K+1)
  all_biases = [None] * (K+1)

  # Initalize parameters for input and output layers
  all_weights[0] = np.random.normal(size=(D, D_i))*np.sqrt(sigma_sq_omega)
  all_weights[-1] = np.random.normal(size=(D_o, D)) * np.sqrt(sigma_sq_omega)
  all_biases[0] = np.random.normal(size=(D,1))* np.sqrt(sigma_sq_omega)
  all_biases[-1]= np.random.normal(size=(D_o,1))* np.sqrt(sigma_sq_omega)

  # Initialize intermediate hidden layers
  for layer in range(1,K):
    all_weights[layer] = np.random.normal(size=(D,D))*np.sqrt(sigma_sq_omega)
    all_biases[layer] = np.random.normal(size=(D,1))* np.sqrt(sigma_sq_omega)

  return all_weights, all_biases

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

# Define the Forward Pass Function
def forward_pass(net_input, all_weights, all_biases):

  # Retrieve number of layers
  K = len(all_weights) -1

  # Define list to store the pre-activations and activations of each layer
  all_f = [None] * (K+1) # Pre-activatins
  all_h = [None] * (K+1) # Activations
  all_h[0] = net_input

  # Calculate activations and pre-activations for all layers
  for layer in range(K):
      # Update preactivations and activations at this layer
      all_f[layer] = all_biases[layer] + np.matmul(all_weights[layer], all_h[layer])
      all_h[layer+1] = ReLU(all_f[layer])

  # Compute the output from the last hidden layer
  all_f[K] = all_biases[K] + np.matmul(all_weights[K], all_h[K])

  # Retrieve the output
  net_output = all_f[K]

  return net_output, all_f, all_h

# Define an indicator function
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>=0] = 1
  x_in[x_in<0] = 0
  return x_in

# Define the backward pass (backpropagation)
def calc_input_output_gradient(x_in, all_weights, all_biases):

  # Compute the forward pass
  y, all_f, all_h = forward_pass(x_in, all_weights, all_biases)

  # Define lists to store the derivatives dl_dweights and dl_dbiases
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)

  # Define list to the derivatives of the loss with respect to the activation and preactivations 
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)
  
  # Compute derivatives of net output with respect to loss
  all_dl_df[K] = np.ones_like(all_f[K])

  # Compute backpropagation
  for layer in range(K,-1,-1):
    all_dl_dbiases[layer] = np.array(all_dl_df[layer])
    all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].transpose())

    all_dl_dh[layer] = np.matmul(all_weights[layer].transpose(), all_dl_df[layer])

    if layer > 0:
      all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * all_dl_dh[layer]


  return all_dl_dh[0],y

# Initialize hyperparameters
D = 200; K = 3

# Initialize parameters
all_weights, all_biases = init_params(K,D)

# Define input value
x = np.ones((1,1))

# Compute the gradient of the neural network
dydx,y = calc_input_output_gradient(x, all_weights, all_biases)

# Compute gradients alternatively using finite differences
delta = 0.00000001
x1 = x
y1,*_ = forward_pass(x1, all_weights, all_biases)
x2 = x+delta
y2,*_ = forward_pass(x2, all_weights, all_biases)
# Finite difference calculation
dydx_fd = (y2-y1)/delta

print("Gradient calculation=%f, Finite difference gradient=%f"%(dydx.squeeze(),dydx_fd.squeeze()))

# Define a plotting function to plot the gradients with respect to input value
def plot_derivatives(K, D):
  # Initialize parameters
  all_weights, all_biases = init_params(K,D)

  x_in = np.arange(-2,2, 4.0/256.0)
  x_in = np.resize(x_in, (1,len(x_in)))
  dydx,y = calc_input_output_gradient(x_in, all_weights, all_biases)

  fig,ax = plt.subplots()
  ax.plot(np.squeeze(x_in), np.squeeze(dydx), 'b-')
  ax.set_xlim(-2,2)
  ax.set_xlabel(r'Input, x')
  ax.set_ylabel(r'Gradient, dy/dx')
  ax.set_title('No layers = %d'%(K))
  plt.show()

# Define a model with 1 hidden layer and 200 neurons and plot gradient
D = 200; K = 1
plot_derivatives(K,D)

# Define a model with 24 hidden layer and 200 neurons and plot gradient
K = 24; D = 200
plot_derivatives(K, D)

# Define a model with 50 hidden layer and 200 neurons and plot gradient
K = 50; D = 200
plot_derivatives(K, D)

# Define an autocorrelation function, which determines how correlated the gradients of the neural network are
def autocorr(dydx):
    ac = np.correlate(dydx, dydx, mode='same')

    return ac

# Define a plotting function to plot the autocorrelation function
def plot_autocorr(K, D):

  # Initialize parameters
  all_weights, all_biases = init_params(K,D)

  x_in = np.arange(-2.0,2.0, 4.0/256)
  x_in = np.resize(x_in, (1,len(x_in)))
  dydx,y = calc_input_output_gradient(x_in, all_weights, all_biases)
  ac = autocorr(np.squeeze(dydx))
  ac = ac / ac[128]

  y = ac[128:]
  x = np.squeeze(x_in)[128:]
  fig,ax = plt.subplots()
  ax.plot(x,y, 'b-')
  ax.set_xlim([0,2])
  ax.set_xlabel('Distance')
  ax.set_ylabel('Autocorrelation')
  ax.set_title('No layers = %d'%(K))
  plt.show()

# Compute the autocorrelation for 200 neurons and 2 hidden layers and plot
D = 200; K =2
plot_autocorr(K,D)

# Compute the autocorrelation for 200 neurons and 50 hidden layers and plot
D = 200; K =50
plot_autocorr(K,D)