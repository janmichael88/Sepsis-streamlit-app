import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import pandas as pd

class InstantiateModel(object):

	def __init__(self):
		def softmax(x, axis=1):

		    ndim = K.ndim(x)
		    if ndim == 2:
		        return K.softmax(x)
		    elif ndim > 2:
		        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
		        s = K.sum(e, axis=axis, keepdims=True)
		        return e / s
		    else:
		        raise ValueError('Cannot apply softmax to a tensor that is 1D')

		#constants for the Model
		self.samples = 325252
		self.Tx = 36
		self.Ty = 36
		self.input = 41
		self.output = 2
		self.n_a = 32
		self.n_s = 64

		#constantlayers
		self.repeator = RepeatVector(self.Tx)
		self.concatenator = Concatenate(axis=-1)
		self.densor1 = Dense(10, activation = "tanh")
		self.densor2 = Dense(1, activation = "relu")
		self.activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
		self.dotor = Dot(axes = 1)
		self.post_activation_LSTM_cell = LSTM(self.n_s, return_state = True)
		self.output_layer = Dense(self.output, activation=softmax)

		#allocate for fitted loadedmode
		self.loaded_model = None
	def one_step_attention(self,a, s_prev):
	    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
	    s_prev = self.repeator(s_prev)
	    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
	    concat = self.concatenator([a, s_prev])
	    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
	    e = self.densor1(concat)
	    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
	    energies = self.densor2(e)
	    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
	    alphas = self.activator(energies)
	    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
	    context = self.dotor([alphas, a])
	    return context

	def create_arch(self):
	    # Define the inputs of your model with a shape (Tx,)
	    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
	    X = Input(shape=(self.Tx, self.input))
	    s0 = Input(shape=(self.n_s,), name='s0')
	    c0 = Input(shape=(self.n_s,), name='c0')
	    s = s0
	    c = c0
	    # Initialize empty list of outputs
	    outputs = []
	    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
	    a = Bidirectional(LSTM(self.n_a, return_sequences = True))(X)    
	    # Step 2: Iterate for Ty steps
	    for t in range(self.Ty):
	        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
	        context = self.one_step_attention(a, s)
	        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
	        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
	        s, _, c = self.post_activation_LSTM_cell(context, initial_state = [s, c])
	        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
	        out = self.output_layer(s)
	        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
	        outputs.append(out)
	    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
	    model = Model(inputs = [X, s0, c0], outputs = outputs)
	    
	    self.loaded_model =  model

	def input_weights(self,path):
		#load in h5 into self.loaded_model vars
		self.loaded_model.load_weights(path)

	def get_predictions(self,Xs):
		#create zero vectors for context and queier
		s0 = np.zeros((1,64))
		c0 = np.zeros((1,64))
		preds = self.loaded_model.predict([Xs, s0,c0])
		probas = []
		for i in range(0,len(preds)):
			probas.append(preds[i][0][1])
		return probas

'''
foo = InstantiateModel()
foo.create_arch()
foo.input_weights('LSTM_finaloutput_v4.h5')
test = pd.read_csv('DemoTestSepsis.csv',index_col=0)
Xs = test.iloc[:,1:-1].values.reshape((1,36,41))
print(foo.get_predictions(Xs))
'''


'''
m = 325252
n_a = 32
n_s = 64
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))

#prediction functions
def return_predictions(model,input):
  preds = model.predict([input,s0[1:2,:],c0[1:2,:]])
  probas = []
  for i in range(0,len(preds)):
    probas.append(preds[i][0][1])
  return probas

	
test = pd.read_csv('DemoTestSepsis.csv',index_col=0)
foo = InstantiateModel()
model = foo.model()
model.load_weights('LSTM_finaloutput_v4.h5')
print(return_predictions(model,test.iloc[:,1:-1].values.reshape((1,36,41))))
'''

