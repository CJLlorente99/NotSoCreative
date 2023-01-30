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
	# It can be that the file is empty or non-existant
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
	storageClient = Client(project=projectName)
	bucket = storageClient.bucket(bucketName)
	blob = bucket.blob(objectNameEmailsDf)
	try:
		contents = StringIO(blob.download_as_string().decode('utf-8'))
		df = pd.read_csv(contents, index_col=['email'])
	except:
		df = pd.DataFrame()
	return df

def subscribeBlobEmailsDf(name: str, email: str, typeOfDigest: str):
	# If the email is present change type and name
	dfEmails = readBlobEmailsDf()
	if len(dfEmails) != 0 and (dfEmails.index == email).any():
		dfEmails.loc[email]['type'] = typeOfDigest
		dfEmails.loc[email]['name'] = name
	else:
		aux = pd.DataFrame({'name': name, 'type': typeOfDigest}, index=[email])
		aux.index.name = 'email'
		dfEmails = pd.concat([dfEmails, aux])

	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
	storageClient = Client(project=projectName)
	bucket = storageClient.bucket(bucketName)
	blob = bucket.blob(objectNameEmailsDf)
	blob.upload_from_string(dfEmails.to_csv())

def unsubscribeBlobEmailsDf(name: str, email: str, typeOfDigest: str):
	# If the email is present change type and name
	dfEmails = readBlobEmailsDf()
	try:
		dfEmails = dfEmails.drop(email)
	except:
		pass

	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
	storageClient = Client(project=projectName)
	bucket = storageClient.bucket(bucketName)
	blob = bucket.blob(objectNameEmailsDf)
	blob.upload_from_string(dfEmails.to_csv())