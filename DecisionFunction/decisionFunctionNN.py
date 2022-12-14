from keras import Sequential
from keras.layers import Dense, Input, Activation


class NNDecisionFunction:
	def __init__(self, nInputs):
		self.model = Sequential()
		self.nInputs = nInputs

	def build_model(self):
		self.model.add(Input(shape=(self.nInputs,)))
		self.model.add(Dense(shape=(3,)))
		self.model.add(Activation("tanh"))
		self.model.add(Dense(units=1, activation='linear'))
		self.model.compile(optimizer='adam', loss='mean_squared_error')

	def train_model(self, xArray, yArray):
		self.model.fit(x=xArray, y=yArray)

	def predict(self, x):
		return self.model.predict(x)
