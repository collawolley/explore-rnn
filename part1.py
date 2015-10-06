# We won't worry about initializations for now.
# Most popular are the Glorot and He initializations.
from __future__ import print_function
import numpy as np

input_size = 10
hidden_size = 10

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

#
# rnn params
#
U = np.random.randn(input_size, hidden_size)
W = np.random.randn(hidden_size, hidden_size)
V = np.random.randn(hidden_size, hidden_size)

def rnn(xt, prev_h):
    ht = np.tanh(xt.dot(U) + prev_h.dot(W))
    ot = ht.dot(V)
    return ot, ht

xt = np.random.randn(10,10)
prev_h = np.random.randn(10,10)
rnn(xt, prev_h)

#
# lstm params
#

# forget gate
W_xf = np.random.randn(input_size, hidden_size)
W_hf = np.random.randn(hidden_size, hidden_size)
b_xf = np.ones(hidden_size).reshape(-1, 1)
b_hf = np.ones(hidden_size).reshape(-1, 1)

# input gate
W_xi = np.random.randn(input_size, hidden_size)
W_hi = np.random.randn(hidden_size, hidden_size)
b_xi = np.ones(hidden_size).reshape(-1, 1)
b_hi = np.ones(hidden_size).reshape(-1, 1)

# output gate
W_xo = np.random.randn(input_size, hidden_size)
W_ho = np.random.randn(hidden_size, hidden_size)
b_xo = np.ones(hidden_size).reshape(-1, 1)
b_ho = np.ones(hidden_size).reshape(-1, 1)

# candidate memory state
W_xc = np.random.randn(input_size, hidden_size)
W_hc = np.random.randn(hidden_size, hidden_size)
b_xc = np.ones(hidden_size).reshape(-1, 1)
b_hc = np.ones(hidden_size).reshape(-1, 1)

def lstm(xt, prev_c, prev_h):
	ft = sigmoid(xt.dot(W_xf) + prev_h.dot(W_hf) +  b_xf + b_hf)
	it = sigmoid(xt.dot(W_xi) + prev_h.dot(W_hi) +  b_xi + b_hi)
	ot = sigmoid(xt.dot(W_xo) + prev_h.dot(W_ho) +  b_xo + b_ho)

	candidate_memory = np.tanh(xt.dot(W_xc) + prev_h.dot(W_hc) +  b_xc + b_hc)
	ct = ft * prev_c + it * candidate_memory
	ht = ot * np.tanh(ct)

	# ct and ht are the outputs, we copy ht and send it to the
	# next layer
	return np.copy(ht), ct, ht

xt = np.random.randn(10,10)
prev_h = np.random.randn(10,10)
prev_c = np.random.randn(10,10)
lstm(xt, prev_c, prev_h)

#
# gru params
#

# update gate
W_xz = np.random.randn(input_size, hidden_size)
W_hz = np.random.randn(hidden_size, hidden_size)

# reset gate
W_xr = np.random.randn(input_size, hidden_size)
W_hr = np.random.randn(hidden_size, hidden_size)

# candidate activations
W_xc = np.random.randn(input_size, hidden_size)
W_hc = np.random.randn(hidden_size, hidden_size)

def gru(xt, prev_h):
	zt = sigmoid(xt.dot(W_xz) + prev_h.dot(W_hz))
	rt = sigmoid(xt.dot(W_xr) + prev_h.dot(W_hr))
	candidate_activation = np.tanh(xt.dot(W_xc) + prev_h.dot(W_hc))
	ht = (1. - zt) * prev_h + zt * candidate_activation

	# similar to a lstm we copy ht for the output
	# to the next layer
	return np.copy(ht), ht

xt = np.random.randn(10,10)
prev_h = np.random.randn(10,10)
gru(xt, prev_h)
