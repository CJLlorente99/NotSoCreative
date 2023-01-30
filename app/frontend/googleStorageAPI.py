import pandas as pd
from io import StringIO
from google.cloud.storage import Client
import os
import json

url = 'https://storage.googleapis.com/datascienceii/myData.csv'
projectName = 'datascienceii'
bucketName = 'datascienceii'
objectNameDf = 'myData.csv'
objectNameEmailsDf = 'emails.csv'
credentials = './application_default_credentials.json'


def readBlobDf():
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
	storageClient = Client(project=projectName)
	bucket = storageClient.bucket(bucketName)
	blob = bucket.blob(objectNameDf)
	try:
		contents = StringIO(blob.download_as_string().decode('utf-8'))
		df = pd.read_csv(contents)
	except:
		df = pd.DataFrame()
	return df

def readBlobEmailsDf() -> pd.DataFrame:
	# TODO
	# df has 3 rows |name|email|type
	# It can be that the file is empty or non-existant
	return pd.DataFrame()

def subscribeBlobEmailsDf(name: str, email: str, typeOfDigest: str):
	# TODO
	# Before writing check that the email is not present
	# df has 3 rows |name|email|type
	pass

def unsubscribeBlobEmailsDf(name: str, email: str, typeOfDigest: str):
	# TODO
	# df has 3 rows |name|email|type
	pass