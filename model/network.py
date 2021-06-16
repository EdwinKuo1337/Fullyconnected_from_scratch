from .layer import *

class Network(object):
    def __init__(self):

        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere. 
        self.fc1 = FullyConnected(28*28, 512) ## Just an example.You can alter sample code anywhere. 
        self.fc2 = FullyConnected(512,256)
        self.fc3 = FullyConnected(256,128)
        self.fc4 = FullyConnected(128,10)
        self.relu = ACTIVITY1()
        self.softmax = SoftmaxWithloss()
        self.z1 = np.zeros((512, 250))
        self.z2 = np.zeros((256, 250))
        self.z3 = np.zeros((128, 250))
        


    def forward(self, x, target):
        x = x.T #(28*28, 32)
        h1 = self.fc1.forward(x) #(512,32)
        ## by yourself .Finish your own NN framework
        z1 = self.relu.forward(h1) #(512,32)
        self.z1 = z1
        h2 = self.fc2.forward(z1) #(256,32)
        z2 = self.relu.forward(h2)
        self.z2 = z2
        h3 = self.fc3.forward(z2) #(128,32)
        z3 = self.relu.forward(h3)
        self.z3 = z3
        h4 = self.fc4.forward(z3)
        
        pred, loss = self.softmax.forward(h4, target)
        
        return pred, loss

    def backward(self):
        ## by yourself .Finish your own NN framework
        dZ4 = self.softmax.backward() #(10, 32)
#         print("dZ2.shape = ", dZ2.shape)
        W3 = self.fc4.backward(dZ4) #(10,128)
#         print("W.shape = ", W.shape)
        
        dZ3 = np.matmul(W3.T, dZ4) * self.relu.backward(self.z3) #(128, 10) * (10, 32) * (128, 32) = (128, 32)
        
        W2 = self.fc3.backward(dZ3) #(128,256)
        dZ2 = np.matmul(W2.T, dZ3) * self.relu.backward(self.z2) #(256,32)
        
        W1 = self.fc2.backward(dZ2) #(256,512)
        dZ1 = np.matmul(W1.T, dZ2) * self.relu.backward(self.z1) #(512,32)
        
        W = self.fc1.backward(dZ1) #(512,28*28)

    def update(self, lr):
        ## by yourself .Finish your own NN framework
#         print(self.fc1.bias.shape)
#         print((self.fc1.bias_grad).shape)
        self.fc1.weight -= self.fc1.weight_grad * lr
        self.fc1.bias -= self.fc1.bias_grad * lr
        self.fc2.weight -= self.fc2.weight_grad * lr
        self.fc2.bias -= self.fc2.bias_grad * lr
        self.fc3.weight -= self.fc3.weight_grad * lr
        self.fc3.bias -= self.fc3.bias_grad * lr
        self.fc4.weight -= self.fc4.weight_grad * lr
        self.fc4.bias -= self.fc4.bias_grad * lr
        
        
