class DenseLayer:
  def __init__(self, features, nth_neurons):
    # initialize weight and bias
    self.weights = 0.01 * torch.rand(features, nth_neurons)
    self.bias = torch.zeros((1, nth_neurons))
  
  # forward pass
  def forward(self, inputs):
    # calculate output values from inputs, weights and biases
    self.output = torch.matmul(inputs, self.weights) + self.bias
