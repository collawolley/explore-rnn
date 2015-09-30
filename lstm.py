from __future__ import print_function
import cgt
import numpy as np
from cgt.nn import parameter, init_array, Constant, HeUniform

class LSTMCell(object):
    def __init__(self, input_size, rnn_size, name="", weight_init=HeUniform(1.0)):
        """
        lstm cell
        """
        # TODO: add bias

        # forget gate weights
        self.W_xf = parameter(init_array(weight_init, (input_size, rnn_size)), name=name+".W_xf")
        self.W_hf = parameter(init_array(weight_init, (rnn_size, rnn_size)), name=name+"W_hf")

        # input gate weights
        self.W_xi = parameter(init_array(weight_init, (input_size, rnn_size)), name=name+".W_xi")
        self.W_hi = parameter(init_array(weight_init, (rnn_size, rnn_size)), name=name+"W_hi")

        # output gate weights
        self.W_xo = parameter(init_array(weight_init, (input_size, rnn_size)), name=name+".W_xo")
        self.W_ho = parameter(init_array(weight_init, (rnn_size, rnn_size)), name=name+"W_ho")

        # candidate value weights
        self.W_xc = parameter(init_array(weight_init, (input_size, rnn_size)), name=name+".W_xc")
        self.W_hc = parameter(init_array(weight_init, (rnn_size, rnn_size)), name=name+"W_hc")



    def __call__(self, x, prev_c, prev_h):
        """
        x is the input
        prev_h is the previous timestep
        prev_c is the previous memory context

        Returns (next_c, next_h).

        next_h should be cloned since it's feed into the next layer and
        the next timstep.
        """

        forget_gate = cgt.sigmoid(x.dot(self.W_xf) + prev_h.dot(self.W_hf))
        input_gate = cgt.sigmoid(x.dot(self.W_xi) + prev_h.dot(self.W_hi))
        output_gate = cgt.sigmoid(x.dot(self.W_xo) + prev_h.dot(self.W_ho))
        candidate_values = cgt.tanh(x.dot(self.W_xc) + prev_h.dot(self.W_hc))

        # new cell state
        next_c = forget_gate * prev_c + input_gate * candidate_values
        # input for next timestep
        next_h = output_gate * cgt.tanh(next_c)

        # NOTE: we feed next_h into the next layer and the next timestep
        # so we should clone the next_h output.
        return next_c, next_h

# Make sure it compiles!

x = cgt.matrix() # (batch_size, n_features)
h = cgt.matrix() # this will later be the identity matrix
c = cgt.matrix() # this will later be the identity matrix

next_c, next_h = LSTMCell(5, 10)(x, c, h)
print("Next Cell State:", next_c, cgt.infer_shape(next_c))
print("Next Hidden:", next_h, cgt.infer_shape(next_h))
