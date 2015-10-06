from __future__ import print_function
import cgt
import numpy as np
from cgt.nn import parameter, init_array, HeUniform

class RNNCell(object):
    def __init__(self, input_size, hidden_size, name="", weight_init=HeUniform(1.0)):
        """
        Initialize an RNN cell
        """

        # input to hidden
        self.W_xh = parameter(init_array(weight_init, (input_size, hidden_size)),
            name=name+".W_xh")

        # hidden to hidden
        self.W_hh = parameter(init_array(weight_init, (hidden_size, hidden_size)),
            name=name+".W_hh")

        # hidden to output
        self.W_ho = parameter(init_array(weight_init, (hidden_size, hidden_size)),
            name=name+".W_ho")

    def __call__(self, x, prev_h):
        """
        x is the input
        prev_h is the input from the previous timestep

        Returns (out, next_h). Feed out into the next layer and
        next_h to the next timestep.
        """

        next_h = cgt.tanh(prev_h.dot(self.W_hh) + x.dot(self.W_xh))
        out = next_h.dot(self.W_ho)
        return out, next_h

# Make sure it compiles!

x = cgt.matrix() # (batch_size, n_features)
h = cgt.matrix() # this will later be the identity matrix

o, next_h = RNNCell(5, 10)(x, h)
print("Output:", o, cgt.infer_shape(o))
print("Next Hidden:", next_h, cgt.infer_shape(next_h))
