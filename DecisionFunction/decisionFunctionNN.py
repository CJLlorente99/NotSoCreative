from keras import Sequential
from keras.layers import Dense, Input, Activation
from keras import metrics
import tensorflow as tf


class NNDecisionFunction:
	def __init__(self, nInputs):
		self.model = Sequential()
		self.nInputs = nInputs
		self.build_model()

	def build_model(self):
		self.model.add(Input(self.nInputs,))
		self.model.add(Dense(5))
		self.model.add(Activation("tanh"))
		self.model.add(Dense(5))
		self.model.add(Activation("tanh"))
		self.model.add(Dense(1, activation='linear'))
		self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=[metrics.mean_squared_error])

	def summary(self):
		print(self.model.summary())

	def train_model(self, xArray, yArray):
		self.model.fit(x=xArray, y=yArray, epochs=100, batch_size=32)

	def predict(self, x):
		return self.model.predict(x)
