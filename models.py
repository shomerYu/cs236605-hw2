import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======

        blocks.append(Linear(in_features, hidden_features[0]))
        if activation == 'relu':
            blocks.append(ReLU())
        else:
            blocks.append(Sigmoid())
        if dropout != 0:
            blocks.append(Dropout(dropout))
        for i in range(len(hidden_features) - 1):
            blocks.append(Linear(hidden_features[i], hidden_features[i + 1]))
            if activation == 'relu':
                blocks.append(ReLU())
            else:
                blocks.append(Sigmoid())
            if dropout != 0:
                blocks.append(Dropout(dropout))
        blocks.append(Linear(hidden_features[-1], num_classes))
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======

        # for simplicity we do the first iteration manually
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels, self.filters[0], kernel_size=3, stride=1, padding=1), nn.ReLU()))
        for k in range(1, self.pool_every):
            layers.append(nn.Sequential(nn.Conv2d(self.filters[k - 1], self.filters[k], 3, 1, 1), nn.ReLU()))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=4,padding=-1))

        # now the rest of the iterations

        for j in range(1, int(len(self.filters) / self.pool_every)):
            for i in range(self.pool_every):
                layers.append(nn.Sequential(
                    nn.Conv2d(self.filters[(j*self.pool_every)+i-1], self.filters[(self.pool_every * j)+i],
                              kernel_size=3, stride=1, padding=1),
                    nn.ReLU()))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=4, padding=-1))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        # now we know that our input is of size filters[last] X in_h*N/2P X in_w*N/2P
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        print(in_channels,in_h,in_w)

        in_channels = int(self.filters[-1])
        in_h = int(self.in_size[1]*(0.25**(len(self.filters) / self.pool_every)))
        in_w = int(self.in_size[2]*(0.25**(len(self.filters) / self.pool_every)))

        newTuple = (in_channels,in_h,in_w)
        mul = newTuple[0] * newTuple[1] * newTuple[2]
        print("val = ")
        print(mul)
        layers.append(nn.Sequential(nn.Linear(mul, self.hidden_dims[0]), nn.ReLU()))
        for i in range(1, len(self.hidden_dims)):
            layers.append(nn.Sequential(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]), nn.ReLU()))
        layers.append(
            nn.Sequential(nn.Linear(self.hidden_dims[-1], self.out_classes), nn.ReLU()))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        seq = self._make_feature_extractor()
        for layer in seq:
            x = layer(x)
            print("ok1")
        layers = self._make_classifier()
        for layer1 in layers:
            x = layer1(x)
            print("ok2")
        out = x
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
