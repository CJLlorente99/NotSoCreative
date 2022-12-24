from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
import graphviz
import pickle

"""
This script contains a class that represents a decision tree classifier that could be use as a decision function.
This is, the inputs given to this model could be TI's/LSTM/Other
"""

class DecisionFunctionTree:
	def __init__(self, max_depth=None, min_samples=None, criterion=None, predictors_list=None):
		if max_depth and min_samples and criterion and predictors_list:
			self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples)
			self.predictors_list = predictors_list
		else:
			pass

	def train(self, x, y):
		self.model = self.model.fit(x, y)

	def show(self):
		dot_data = tree.export_graphviz(self.model, out_file=None,filled=True,feature_names=self.predictors_list)
		graphviz.Source(dot_data).view()

	def save(self, filename):
		pickle.dump(self.model, open(filename, 'wb'))

	def load(self, filename):
		self.model = pickle.load(open(filename, 'rb'))

	def predict_test(self, x, y_true, print_report):
		y = self.model.predict(x)
		if print_report:
			report = classification_report(y_true, y)
			print(report)

		return y

	def predict(self, x):
		return self.model.predict(x)
