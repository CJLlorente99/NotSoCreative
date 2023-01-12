from dailyStrategy import DailyStrategy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras import initializers
from sklearn.preprocessing import StandardScaler
import numpy as np

class LSTMEnsemble1(DailyStrategy):
	def possiblyOperationMorning(self, data):
		self.perToInvest = getPredictionLSTM(data)

	def possiblyOperationAfternoon(self, data) :
		self.perToInvest = -1  # Always sell all


def getPredictionLSTM(data):

	data['Target'] = data['Return_intraday'].shift(-1)
	y_target = np.asarray([1 if data.Target[i] > 0 else 0 for i in range(len(data))]).reshape(-1, 1)
	data = data.drop(['Target'], axis=1)

	scaler = StandardScaler()
	data_set_scaled = scaler.fit_transform(data)
	data_set_scaled = np.concatenate((data_set_scaled[1:], y_target[1:]), axis=1)

	backcandles = 30

	# choose columns: all but target variable (its last column)
	liste = list(range(0, data.shape[1] - 1))

	# print(data.iloc[:, liste])

	# split data into train test sets
	pred_days = 1

	# prepare data for lstm
	X_train, X_test, y_train, y_test = prepare_data(data_set_scaled, backcandles, liste, pred_days)

	# train and predict
	# n_members -> how many predictors we wanted to use
	n_members = 10
	epochs = 32
	batch_size = 8
	ensemble, y_pred, prob = fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size)

	# majority vote, probs for amount
	return majority_vote(y_pred)[0]

def class_LSTM(n_inputs, n_features):
	model = Sequential()
	model.add(LSTM(units=197, return_sequences=True, bias_initializer=initializers.Constant(0.01), input_shape=(n_inputs, n_features)))
	model.add(Dropout(0.1))
	model.add(LSTM(178))
	model.add(Dropout(0.1))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
	return model

def prepare_data(data_set_scaled, backcandles, liste, pred_days):
	X = []
	for j in range(len(liste)):  # data_set_scaled[0].size):#2 columns are target not X
		X.append([])
		for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
			X[j].append(data_set_scaled[i - backcandles:i, liste[j]])

	# move axis from 0 to position 2
	X = np.moveaxis(X, [0], [2])

	X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
	y = np.reshape(yi, (len(yi), 1))
	splitlimit = X.shape[0] - pred_days

	X_train, X_test = X[:splitlimit], X[splitlimit:]
	y_train, y_test = y[:splitlimit], y[splitlimit:]


	return X_train, X_test, y_train, y_test

def fit_model(X_train, y_train, epochs, batch_size):
	# define neural network model
	n_inputs = X_train.shape[1]
	n_features = X_train.shape[2]
	model = class_LSTM(n_inputs, n_features)
	# fit the model on the training dataset
	early_stopping = EarlyStopping(monitor="accuracy", patience=10, mode='auto', min_delta=0)
	model.fit(X_train, y_train, verbose=2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
	return model

def fit_ensemble(n_members, X_train, X_test, y_train, y_test, epochs, batch_size):
	ensemble = list()
	y_pred = np.empty((n_members, y_test.shape[0]))
	prob = np.empty((n_members, y_test.shape[0]))
	for i in range(n_members):
		# define and fit the model on the training set
		model = fit_model(X_train, y_train, epochs, batch_size)
		# evaluate model on the test set
		yhat = model.predict(X_test, verbose=2)
		prob[i, :] = yhat.flatten()

		yhat[yhat > 0.5] = 1
		yhat[yhat <= 0.5] = 0

		# store the model and prediction
		ensemble.append(model)
		y_pred[i, :] = yhat.flatten()
	return ensemble, y_pred, prob

def majority_vote(yhat):
	# if majority is 1 -> signal = 1 -> buy
	y_mean = []
	probs = []
	for i in range(yhat.shape[1]):
		y_10 = yhat[:, i]
		probs.append(np.sum(y_10))
		n_one = np.count_nonzero(y_10 == 1)
		length = round(y_10.shape[0] * 0.5)
		if n_one >= length:
			y_mean.append(1)
		else:
			y_mean.append(-1)
	#error here
	#probs_mean = probs / len(probs)
	return y_mean #, probs_mean