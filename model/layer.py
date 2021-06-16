import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features,in_features) * 0.01
        self.bias = np.zeros((out_features, 1))
        self.in_mat = np.zeros((in_features, 250))
        self.weight_grad = np.zeros((out_features, in_features))
        self.bias_grad = np.zeros((out_features, 1))
        
        
    def forward(self, x):
        self.in_mat = x
        out = np.matmul(self.weight, x) + self.bias
        
        return out

    def backward(self, output_grad):
        m, _ = self.weight.shape
        input_grad = self.weight #(out_f, in_f)
        self.weight_grad = 1 / m * np.matmul(output_grad, self.in_mat.T)
#         print(output_grad.shape)
        self.bias_grad = 1 / m * np.sum(output_grad, axis = 1)
        self.bias_grad = self.bias_grad.reshape(self.out_features, 1)
#         print(self.bias_grad.shape)

        return input_grad

## ReLU
class ACTIVITY1(_Layer):
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return x > 0

class SoftmaxWithloss(_Layer):
    def __init__(self):
        self.loss = np.zeros((10,100))

    def forward(self, x, target):
        

        '''Softmax'''
        c, b = x.shape
        predict = np.zeros((c,b))
        for i in range(b):
            sm = 0
            for j in range(c):
                sm += np.exp(x[j][i])
            for j in range(c):
                predict[j][i] = np.exp(x[j][i])/sm
#         for i in range(b):
#             for j in range(c)
#             predict[:][i] = np.exp(x[:][i]) / np.sum(np.exp(x[:][i]))

        

        '''Average loss'''
        self.loss = predict - target.T #(10, 32)
        
        your_loss = np.sum(np.abs(self.loss))

        return predict.T, your_loss

    def backward(self):
        input_grad = self.loss

        return input_grad
    
    