import pandas as pd
from io import StringIO
from google.cloud.storage import Client
import os
import json

url = 'https://storage.googleapis.com/datascienceii/myData.csv'
projectName = 'datascienceii'
bucketName = 'datascienceii'
objectNameDf = 'myData.csv'
credentials = '/home/carlosllocor/NotSoCreative/app/service/application_default_credentials.json'


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
