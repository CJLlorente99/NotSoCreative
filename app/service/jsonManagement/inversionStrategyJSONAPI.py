import json
import os.path


class InputParameter:
	def __init__(self, name, value):
		self.name = name
		self.value = value

	def asDict(self):
		return {'Name': self.name, 'Value': self.value}

class StrategyInput:
	def __init__(self, name: str, description: str, dfName: str, keyName: str, listParameters: [InputParameter]):
		self.name = name
		self.description = description
		self.dfName = dfName
		self.keyName = keyName
		self.listParameters = listParameters

	def asDict(self):
		res = {'Name': self.name, 'Description': self.description, 'DfName': self.dfName, 'Key': self.keyName}
		aux = []
		for param in self.listParameters:
			aux.append(param.asDict())
		res['Parameters'] = aux

		return res

class Strategy:
	def __init__(self, name: str, description: str, listInputs: [StrategyInput]):
		self.name = name
		self.initialMoney = 10000
		self.description = description
		self.listInputs = listInputs

	def asDict(self):
		res = {'Name': self.name, 'Description': self.description}
		aux = []
		for inp in self.listInputs:
			aux.append(inp.asDict())
		res['Inputs'] = aux

		return res

	def getListDfNameInputs(self):
		res = []
		for inp in self.listInputs:
			res.append(inp.dfName)

		return res


class JsonStrategyManager:
	def __init__(self, jsonRawData):
		self.jsonRawData = json.load(jsonRawData)

	def readAllStrategies(self):
		jsonData = json.load(self.jsonRawData)

		entries = []
		for entry in jsonData:
			entries.append(json.loads(entry))

		for entry in entries:
			print('Name of the strategy')
			print("\t" + entry['Name'])
			print('Description')
			print("\t" + entry['Description'])
			print('Description of the different inputs')
			for inp in entry['Inputs']:
				print("\t" + inp['Name'])
				print("\t" + inp['DfName'])
				print("\t" + inp['Key'])
				print("\t" + inp['Description'])
				print("\t" + 'Description of the different parameters')
				for param in inp['Parameters']:
					print("\t\t" + param['Name'])
					print("\t\t\t" + str(param['Value']))

	def listStrategies(self):

		entries = []
		for entry in self.jsonRawData:
			entries.append(json.loads(entry))

		strategies = []
		for entry in entries:
			aux = Strategy(entry['Name'], entry['Description'], self.listInput(entry['Inputs']))
			strategies.append(aux)

		return strategies

	def listInputs(self):

		entries = []
		for entry in self.jsonRawData:
			entries.append(json.loads(entry))

		inputs = []
		for entry in entries:
			for inp in entry['Inputs']:
				aux = StrategyInput(inp['Name'], inp['Description'], inp['DfName'], inp['Key'], self.listParameters(inp['Parameters']))
				inputs.append(aux)

		return inputs


	def listInput(self, listDictInputs):
		inputs = []
		for inp in listDictInputs:
			aux = StrategyInput(inp['Name'], inp['Description'], inp['DfName'], inp['Key'], self.listParameters(inp['Parameters']))
			inputs.append(aux)

		return inputs

	def listParameters(self, listDictParameters):
		params = []
		for param in listDictParameters:
			aux = InputParameter(param['Name'], param['Value'])
			params.append(aux)

		return params
