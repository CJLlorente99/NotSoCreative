import pandas as pd
from io import StringIO
from google.cloud.storage import Client
import os
import json

url = 'https://storage.googleapis.com/datascienceii/myData.csv'
projectName = 'datascienceii'
bucketName = 'datascienceii'
objectNameDf = 'myData.csv'
objectNameJson = 'strategies.json'
credentials = 'C:/Users/carlo/AppData/Roaming/gcloud/application_default_credentials.json'


def readBlobDf():
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
	storageClient = Client(project=projectName)
	bucket = storageClient.bucket(bucketName)
	blob = bucket.blob(objectNameDf)
	try:
		contents = StringIO(blob.download_as_string().decode('utf-8'))
		df = pd.read_csv(contents, index_col=['Date'])
	except:
		df = pd.DataFrame()
	return df

def updateBlobDf(df):
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
	storageClient = Client(project=projectName)
	bucket = storageClient.bucket(bucketName)
	blob = bucket.blob(objectNameDf)
	blob.upload_from_string(df.to_csv())

def readBlobJson():
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
	storageClient = Client(project=projectName)
	bucket = storageClient.bucket(bucketName)
	blob = bucket.blob(objectNameJson)
	contents = StringIO(blob.download_as_string().decode('utf-8'))
	return contents

def updateBlobJson(jsonData):
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
	storageClient = Client(project=projectName)
	bucket = storageClient.bucket(bucketName)
	blob = bucket.blob(objectNameJson)
	blob.upload_from_string(jsonData)