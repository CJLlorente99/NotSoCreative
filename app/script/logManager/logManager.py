from datetime import datetime

class LogManager:
	def __init__(self, fileName):
		self.fileName = fileName

	def writeLog(self, severity, msg):
		string = f'{datetime.now()} -> [{severity}] {msg}\n'
		with open(self.fileName, mode='a') as f:
			f.write(string)
			f.close()
