from dailyStrategy import DailyStrategy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM, Bidirectional
import numpy as np
from keras import initializers

class BiLSTMWindowRobMMT1T2Legacy(DailyStrategy):
	def possiblyOperationMorning(self, data):
		res = self.calculatePrediction(data)
		print(f'BiLSTMWindowRobMMT1T2Legacy res {res}')
		if self.nonInvestedMoney == 0:
			x = 0
		else:
			x = 5000 / self.nonInvestedMoney
		if res >= 0:
			if x > 1:
				self.perToInvest = 1
			else:
				self.perToInvest = x
		else:
			self.perToInvest = res

	def possiblyOperationAfternoon(self, data):
		self.perToInvest = 0

	def calculatePrediction(self, data):
		# data is already shifted when it comes here
		res = data[data.columns]

		df_log = np.sqrt(np.log(data.Open))
		df_log_diff = df_log - df_log.shift()
		res['Return_Open'] = df_log_diff
		res['Target'] = df_log_diff
		res.dropna(inplace=True)

		# only for
		pred_days = 3

		# how many days i want to predict
		step_out = 3

		# how many last days i include in my prediction
		backcandles = 8  # 10

		# to save decision and predictions
		y_predictions = []

		# days to predict
		test_days = step_out

		# scale data with robust
		scaler_r = RobustScaler()
		data_set_scaled = scaler_r.fit_transform(res)
		# scale data
		scaler_m = MinMaxScaler()
		data_set_scaled = scaler_m.fit_transform(data_set_scaled)

		# prepare data for lstm
		data_set_scaled = np.vstack([data_set_scaled, np.zeros((test_days, data_set_scaled.shape[1]))])
		X_train, X_test, y_train, y_test = prepare_multidata(data_set_scaled, backcandles, pred_days, test_days)

		n_members = 5
		epochs = 36  #
		batch_size = 2
		ensemble, y_pred_scale = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs,
											   batch_size)

		# inverse scaling
		y_pred_scale = inverse_scaling(res, y_pred_scale, scaler_m)

		y_pred = inverse_scaling(res, y_pred_scale, scaler_r)

		# bounds, mean -> further I only use mean
		lower, y_mean, upper = calculate_bounds(y_pred)

		# inverse transformation: not possible to use for real script -> because of index, in real script we have no test_or
		# so for real script it might be better to use not a index or so
		y_mean = pd.DataFrame(y_mean, range(len(y_mean)), columns=['Open'])
		y_mean['Open'] = y_mean['Open'] + df_log.shift().values[-test_days:]
		y_mean = (y_mean ** 2)
		y_mean = np.exp(y_mean)

		y_predictions.append(np.asarray(y_mean))
		y_mean = np.asarray(y_mean).flatten()
		print(f'BiLSTMWindowRobMMT1T2Legacy {y_mean}, open {data.Open.values[-1]}')

		# build decision rule
		if data.Open.values[-1] < y_mean[1] < y_mean[2]:
			return 1
		elif data.Open.values[-1] < y_mean[2] < y_mean[1]:
			return 0.5
		elif y_mean[1] < data.Open.values[-1] < y_mean[2]:
			return -0.5
		elif y_mean[2] < data.Open.values[-1] and y_mean[1] < data.Open.values[-1]:
			return -1
		elif y_mean[2] < data.Open.values[-1] < y_mean[1]:
			return 0
		else:
			return -1

def build_model(n_inputs, n_features, n_outputs):
	opt = Adam(learning_rate=0.000224)
	model = Sequential()
	model.add(Bidirectional(LSTM(units=150, return_sequences=True,  bias_initializer=initializers.Constant(0.01),
				   kernel_initializer='he_uniform', input_shape=(n_inputs, n_features))))
	model.add(Dropout(0.1))
	model.add(Bidirectional(LSTM(units=100)))
	model.add(Dropout(0.1))
	# model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
	model.add(Dense(units=n_outputs, activation='linear'))
	model.compile(optimizer=opt, loss='mse')
	# history = model.fit
	return model

def fit_model(X_train, y_train, epochs, batch_size):
	# define neural network model
	n_inputs = X_train.shape[1]
	n_features = X_train.shape[2]
	n_outputs = y_train.shape[1]
	model = build_model(n_inputs, n_features, n_outputs)
	# model.fit(X_train, y_train, batch_size=10, epochs=93)
	# fit the model on the training dataset
	early_stopping = EarlyStopping(monitor="loss", patience=10, mode='auto', min_delta=0)
	model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, validation_split=0.15, callbacks=[early_stopping])
	return model


def fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size):
	ensemble = list()
	y_pred = np.empty((n_members, y_test.shape[1]))

	for i in range(n_members):
		# define and fit the model on the training set
		model = fit_model(X_train, y_train, epochs, batch_size)

		# evaluate model on the test set
		yhat = model.predict(X_test, verbose=2)
		print(f'BiLSTMWindowRobMMT1T2Legacy yhat {yhat}, member {i}')
		mae = mean_absolute_error(y_test, yhat)
		# store the model and prediction
		ensemble.append(model)
		y_pred[i, :] = yhat.flatten()
	return ensemble, y_pred


def inverse_scaling(data, yhat, scaler):
	y_preds = np.empty((yhat.shape[0], yhat.shape[1]))
	for i in range(yhat.shape[0]):
		y_pred = np.tile(yhat[i, :].reshape(-1, 1), (1, data.shape[1]))
		y_pred = scaler.inverse_transform(y_pred)
		y_pred = y_pred[:, -1]
		y_preds[i, :] = y_pred
	return y_preds



def calculate_bounds(yhat):
	lower = []
	upper = []
	y_mean = []
	for i in range(yhat.shape[1]):
		interval = 1.96 * yhat[:, i].std()
		y_mean.append(yhat[:, i].mean())
		lower.append(yhat[:, i].mean() - interval)
		upper.append(yhat[:, i].mean() + interval)
	return lower, y_mean, upper


def prepare_multidata(data_set_scaled, backcandles, pred_days, step_out):
	# preparing data
	X = list()
	y = list()
	y_t = np.array(data_set_scaled[:, -1])

	for i in range(backcandles, data_set_scaled.shape[0] + 1 - step_out):
		X.append(data_set_scaled[i - backcandles:i, :-1])
		y.append(y_t[i:i + step_out])

	# move axis from 0 to position 2
	X = np.array(X)
	y = np.array(y)

	X = X.reshape(y.shape[0], backcandles, data_set_scaled.shape[1] - 1)

	# only last row because it contains the last step_out values (step_out is a number: first iteration: 10, last 1)
	splitlimit = X.shape[0] - 1
	X_train, X_test = X[:splitlimit], X[splitlimit:]
	y_train, y_test = y[:splitlimit], y[splitlimit:]

	return X_train, X_test, y_train, y_test