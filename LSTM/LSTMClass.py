import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Activation
import numpy as np
from keras.utils import to_categorical
from keras.backend import clear_session

"""
This file contains a class that abstracts the LSTM network
"""

class LSTMClass:
	def __init__(self):
		self.data = pd.DataFrame()
		self.model = None

	def trainAndPredictMorning(self, dataUntilToday):
		self.data = dataUntilToday.copy()
		self.data['Open'] = self.data['Open'].shift(-1)
		self.data = self.data[:-1]
		self.data['RSI'] = ta.momentum.RSIIndicator(self.data.Close, 15, True).rsi()
		self.data['EMAF'] = ta.trend.EMAIndicator(self.data.Close, 20, True).ema_indicator()
		self.data['EMAM'] = ta.trend.EMAIndicator(self.data.Close, 100, True).ema_indicator()
		self.data['EMAS'] = ta.trend.EMAIndicator(self.data.Close, 150, True).ema_indicator()

		self.data["log(Open)"] = np.log(self.data["Open"])
		# Target variable = log(price(t+1)-log(t))
		self.data["Return_before"] = self.data["log(Open)"] - self.data["log(Open)"].shift(+1)
		self.data["Return"] = self.data["log(Open)"].shift(-1) - self.data["log(Open)"]
		# Class: 1 = positive return, 0 = negative return
		self.data["Class"] = [1 if self.data.Return[i] > 0 else 0 for i in range(len(self.data))]
		# data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
		self.data.dropna(inplace=True)
		self.data.reset_index(inplace=True)
		self.data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

		sc = StandardScaler()
		data_set_scaled = sc.fit_transform(self.data)

		X = []
		backcandles = 30
		list = [4, 5, 6, 7, 9]
		for j in range(len(list)):  # data_set_scaled[0].size):#2 columns are target not X
			X.append([])
			for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
				X[j].append(data_set_scaled[i - backcandles:i, list[j]])
		X = np.moveaxis(X, [0], [2])
		X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -2])
		y = np.reshape(yi, (len(yi), 1))

		X_train, X_test = X[:-1], X[-1:]
		y_train, y_test = y[:-1], y[-1:]

		np.random.seed(10)

		# Change nr of inputs
		lstm_input = Input(shape=(backcandles, len(list)), name='lstm_input')
		inputs = LSTM(150, name='first_layer')(lstm_input)
		inputs = Dense(1, name='dense_layer')(inputs)
		output = Activation('linear', name='output')(inputs)
		self.model = Model(inputs=lstm_input, outputs=output)
		adam = optimizers.Adam()
		self.model.compile(optimizer=adam, loss='mse')
		self.model.fit(x=X_train, y=y_train, batch_size=15, epochs=40, shuffle=True, validation_split=0)

		result = self.model.predict(X_test)
		clear_session()
		return result

	def trainAndPredictAfternoon(self, dataUntilToday):
		self.data = dataUntilToday.copy()
		self.data['RSI'] = ta.momentum.RSIIndicator(self.data.Close, 15, True).rsi()
		self.data['EMAF'] = ta.trend.EMAIndicator(self.data.Close, 20, True).ema_indicator()
		self.data['EMAM'] = ta.trend.EMAIndicator(self.data.Close, 100, True).ema_indicator()
		self.data['EMAS'] = ta.trend.EMAIndicator(self.data.Close, 150, True).ema_indicator()

		self.data["log(Open)"] = np.log(self.data["Open"])
		# Target variable = log(price(t+1)-log(t))
		self.data["Return_before"] = self.data["log(Open)"] - self.data["log(Open)"].shift(+1)
		self.data["Return"] = self.data["log(Open)"].shift(-1) - self.data["log(Open)"]
		# Class: 1 = positive return, 0 = negative return
		self.data["Class"] = [1 if self.data.Return[i] > 0 else 0 for i in range(len(self.data))]
		# data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
		self.data.dropna(inplace=True)
		self.data.reset_index(inplace=True)
		self.data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

		sc = StandardScaler()
		data_set_scaled = sc.fit_transform(self.data)

		X = []
		backcandles = 30
		list = [4, 5, 6, 7, 9]
		for j in range(len(list)):  # data_set_scaled[0].size):#2 columns are target not X
			X.append([])
			for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
				X[j].append(data_set_scaled[i - backcandles:i, list[j]])
		X = np.moveaxis(X, [0], [2])
		X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -2])
		y = np.reshape(yi, (len(yi), 1))

		X_train, X_test = X[:-1], X[-1:]
		y_train, y_test = y[:-1], y[-1:]

		np.random.seed(10)

		# Change nr of inputs
		lstm_input = Input(shape=(backcandles, len(list)), name='lstm_input')
		inputs = LSTM(150, name='first_layer')(lstm_input)
		inputs = Dense(1, name='dense_layer')(inputs)
		output = Activation('linear', name='output')(inputs)
		self.model = Model(inputs=lstm_input, outputs=output)
		adam = optimizers.Adam()
		self.model.compile(optimizer=adam, loss='mse')
		self.model.fit(x=X_train, y=y_train, batch_size=15, epochs=40, shuffle=True, validation_split=0)

		result = self.model.predict(X_test)
		clear_session()
		return result

	def trainAndPredictClassificationMorning(self, dataUntilToday):
		self.data = dataUntilToday.copy()
		self.data['Open'] = self.data['Open'].shift(-1)
		self.data = self.data[:-1]
		self.data['RSI'] = ta.momentum.RSIIndicator(self.data.Close, 15, True).rsi()
		self.data['EMAF'] = ta.trend.EMAIndicator(self.data.Close, 20, True).ema_indicator()
		self.data['EMAM'] = ta.trend.EMAIndicator(self.data.Close, 100, True).ema_indicator()
		self.data['EMAS'] = ta.trend.EMAIndicator(self.data.Close, 150, True).ema_indicator()

		self.data["log(Open)"] = np.log(self.data["Open"])
		# Target variable = log(price(t+1)-log(t))
		self.data["Return_before"] = self.data["log(Open)"] - self.data["log(Open)"].shift(+1)
		self.data["Return"] = self.data["log(Open)"].shift(-1) - self.data["log(Open)"]
		# Class: 1 = positive return, 0 = negative return
		self.data["Class"] = [1 if self.data.Return[i] > 0 else 0 for i in range(len(self.data))]
		# data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
		self.data.dropna(inplace=True)
		self.data.reset_index(inplace=True)
		self.data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

		sc = StandardScaler()
		data_set_scaled = sc.fit_transform(self.data)
		data_set_scaled[:, -1] = np.array(self.data["Class"])

		X = []
		backcandles = 30
		list = [4, 5, 6, 7, 11]
		for j in range(len(list)):  # data_set_scaled[0].size):#2 columns are target not X
			X.append([])
			for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
				X[j].append(data_set_scaled[i - backcandles:i, list[j]])
		X = np.moveaxis(X, [0], [2])
		X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
		y = np.reshape(yi, (len(yi), 1))

		X_train, X_test = X[:-1], X[-1:]
		y_train, y_test = y[:-1], y[-1:]
		y_train, y_test = to_categorical(y_train), to_categorical(y_test)

		np.random.seed(10)

		# Change nr of inputs
		lstm_input = Input(shape=(backcandles, len(list)), name='lstm_input')
		inputs = LSTM(100, name='first_layer')(lstm_input)
		inputs = Dense(20, name='second_layer')(inputs)
		inputs = Dense(10, name='dense_layer')(inputs)
		output = Dense(2, name='softmax', activation='softmax')(inputs)
		self.model = Model(inputs=lstm_input, outputs=output)
		adam = optimizers.Adam()
		self.model.compile(optimizer=adam, loss='mse')
		self.model.fit(x=X_train, y=y_train, batch_size=15, epochs=40, shuffle=True, validation_split=0)

		result = self.model.predict(X_test)
		clear_session()
		return result

	def trainAndPredictClassificationAfternoon(self, dataUntilToday):
		self.data = dataUntilToday.copy()
		self.data['RSI'] = ta.momentum.RSIIndicator(self.data.Close, 15, True).rsi()
		self.data['EMAF'] = ta.trend.EMAIndicator(self.data.Close, 20, True).ema_indicator()
		self.data['EMAM'] = ta.trend.EMAIndicator(self.data.Close, 100, True).ema_indicator()
		self.data['EMAS'] = ta.trend.EMAIndicator(self.data.Close, 150, True).ema_indicator()

		self.data["log(Open)"] = np.log(self.data["Open"])
		# Target variable = log(price(t+1)-log(t))
		self.data["Return_before"] = self.data["log(Open)"] - self.data["log(Open)"].shift(+1)
		self.data["Return"] = self.data["log(Open)"].shift(-1) - self.data["log(Open)"]
		# Class: 1 = positive return, 0 = negative return
		self.data["Class"] = [1 if self.data.Return[i] > 0 else 0 for i in range(len(self.data))]
		# data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
		self.data.dropna(inplace=True)
		self.data.reset_index(inplace=True)
		self.data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

		sc = StandardScaler()
		data_set_scaled = sc.fit_transform(self.data)
		data_set_scaled[:, -1] = np.array(self.data["Class"])

		X = []
		backcandles = 30
		list = [4, 5, 6, 7, 11]
		for j in range(len(list)):  # data_set_scaled[0].size):#2 columns are target not X
			X.append([])
			for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
				X[j].append(data_set_scaled[i - backcandles:i, list[j]])
		X = np.moveaxis(X, [0], [2])
		X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
		y = np.reshape(yi, (len(yi), 1))

		X_train, X_test = X[:-1], X[-1:]
		y_train, y_test = y[:-1], y[-1:]
		y_train, y_test = to_categorical(y_train), to_categorical(y_test)

		np.random.seed(10)

		# Change nr of inputs
		lstm_input = Input(shape=(backcandles, len(list)), name='lstm_input')
		inputs = LSTM(100, name='first_layer')(lstm_input)
		inputs = Dense(20, name='second_layer')(inputs)
		inputs = Dense(10, name='dense_layer')(inputs)
		output = Dense(2, name='softmax', activation='softmax')(inputs)
		self.model = Model(inputs=lstm_input, outputs=output)
		adam = optimizers.Adam()
		self.model.compile(optimizer=adam, loss='mse')
		self.model.fit(x=X_train, y=y_train, batch_size=15, epochs=40, shuffle=True, validation_split=0)

		result = self.model.predict(X_test)
		clear_session()
		return result