from __future__ import print_function
import cgt
import numpy as np
from cgt.nn import parameter, init_array, Constant

# ignore bias for the sake of simplicity
class RNNCell(object):
    def __init__(self, input_size, hidden_size, output_size, 
            name="", weight_init=Constant(0)):
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
        self.W_ho = parameter(init_array(weight_init, (hidden_size, output_size)),
                name=name+".W_ho")

    def __call__(self, x, h):
        """
        x is the input
        h is the input from the previous timestep

        Returns (out, next_h). Feed out into the next layer and
        next_h to the next timestep.
        """
        # should this an an elementwise add?
        next_h = cgt.tanh(cgt.add(h.dot(self.W_hh), x.dot(self.W_xh)))
        out = next_h.dot(self.W_ho)
        return out, next_h

x = cgt.matrix()
h = cgt.matrix()

o, next_h = RNNCell(5, 10, 5)(x, h)
print("Output:", o, cgt.infer_shape(o))
print("Next Hidden:", next_h, cgt.infer_shape(next_h))


