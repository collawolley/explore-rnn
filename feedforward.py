from __future__ import print_function
import cgt
import numpy as np
from cgt.nn import parameter, init_array, HeUniform, Constant

# ignore bias for the sake of simplicity
class FeedforwardCell(object):
    def __init__(self, input_size, output_size, name="", 
            weight_init=HeUniform(1.0), bias_init=Constant(0)):
        """
        Initialize an Feedforward cell.
        """

        self.W = parameter(init_array(weight_init, (input_size, output_size)),
                name=name+".W")
        self.b = parameter(init_array(bias_init, (1, output_size)), name=name+'.b')

    def __call__(self, x):
        """
        x is the input

        Returns the output to feed as the input into the next layer.
        """

        return cgt.broadcast("+", x.dot(self.W), self.b, "xx,1x")

# Make sure it compiles!

# x is a matrix of size (batch_size, features_size)
x = cgt.matrix() 
o = FeedforwardCell(5, 10)(x)
print("Output:", o, cgt.infer_shape(o))

