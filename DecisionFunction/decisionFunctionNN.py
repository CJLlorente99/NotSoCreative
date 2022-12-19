from keras import Sequential
from keras.layers import Dense, Input, Activation, Normalization
from keras import metrics
from keras.models import load_model
import tensorflow as tf


class NNDecisionFunction:
	def __init__(self, nInputs=1):
		self.model = Sequential()
		self.nInputs = nInputs
		self.build_model()

	def build_model(self):
		self.model.add(Input(self.nInputs,))
		self.model.add(Dense(20))
		self.model.add(Activation("tanh"))
		self.model.add(Dense(10))
		self.model.add(Activation("tanh"))
		self.model.add(Dense(5))
		self.model.add(Activation("tanh"))
		self.model.add(Dense(1, activation='linear'))
		self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

	def getWeights(self):
		for layer in self.model.layers:
			print(layer.get_weights())

	def summary(self):
		print(self.model.summary())

	def train_model(self, xArray, yArray):
		self.model.fit(x=xArray, y=yArray, epochs=50, batch_size=8)

	def predict(self, x):
		return self.model.predict(x)

	def save(self, name):
		self.model.save("../data/model" + name + ".h5")

	def load(self, fileName):
		self.model = load_model(fileName)

class NNDecisionFunctionClassification:
	def __init__(self, nInputs=1):
		self.model = Sequential()
		self.nInputs = nInputs
		self.build_model()

	def build_model(self):
		self.model.add(Input(self.nInputs,))
		self.model.add(Dense(10))
		self.model.add(Activation("relu"))
		self.model.add(Dense(6))
		self.model.add(Activation("relu"))
		self.model.add(Dense(3))
		self.model.add(Activation("relu"))
		self.model.add(Dense(2, activation='softmax'))
		self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

	def getWeights(self):
		for layer in self.model.layers:
			print(layer.get_weights())

	def summary(self):
		print(self.model.summary())

	def train_model(self, xArray, yArray):
		self.model.fit(x=xArray, y=yArray, epochs=50, batch_size=8)

	def predict(self, x):
		return self.model.predict(x)

	def save(self, name):
		self.model.save("../data/model" + name + ".h5")

	def load(self, fileName):
		self.model = load_model(fileName)
