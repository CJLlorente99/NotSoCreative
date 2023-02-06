import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from renderer import Renderer
import pytz
from datetime import datetime, timedelta
from googleStorageAPI import readBlobEmailsDf, readBlobDf
import yfinance as yf
import time

"""
This file is to be called after the stock market is closed in order to send the daily digest
1) Perform checks (has stock market close already, has stock market opened today)
2) Instantiate a Renderer and render the two types of html strings (long and short)
3) Load email information (emails and type of digest)
4) Send emails
"""

notSoCreativeEmail = 'notsocreative2023@gmail.com'

def main():

	"""
	1) Perform checks (has stock market close already, has stock market opened today)
	"""
	# Stock close already?
	todayDate = datetime.now(pytz.timezone('America/New_York'))
	if todayDate <= todayDate.replace(hour=16, minute=0, second=0):
		return

	# Stock opened today?
	stockData = yf.download('^GSPC', (todayDate-timedelta(days=7)).strftime('%Y-%m-%d'), (todayDate+timedelta(days=1)).strftime('%Y-%m-%d'))
	if stockData.index[-1].to_pydatetime().date() != todayDate.date():
		return

	"""
	2) Instantiate a Renderer
	"""
	rend = Renderer(todayDate)

	"""
	3) Load email information (emails and type of digest)
	"""
	dfEmails = readBlobEmailsDf()

	"""
	4) Send emails
	"""
	# Connect to google gmail server
	s = smtplib.SMTP('smtp.gmail.com', 587)
	s.ehlo()
	s.starttls()
	s.ehlo()
	s.login(notSoCreativeEmail, 'nfkpbmnkifameswu')

	for index, row in dfEmails.iterrows():
		# Render messages
		if row['type'] == 'Short':
			renderedHTML = rend.renderShortDailyDigest(row['name'])
		else:
			renderedHTML = rend.renderLongDailyDigest(row['name'])

		# Create a msg with the different parts
		msg = MIMEMultipart('related')
		msg['Subject'] = '[DataScienceII] Daily Digest ' + todayDate.strftime('%d/%m/%Y')
		msg['From'] = notSoCreativeEmail
		msg['To'] = index
		msg.attach(MIMEText(renderedHTML, 'html'))

		fp = open('/home/carlosllocor/NotSoCreative/app/dailyDigestAutomatization/png.png', 'rb')
		msgImage = MIMEImage(fp.read())
		fp.close()
		msgImage.add_header('Content-ID', '<logoNotSoCreative>')
		msg.attach(msgImage)

		# Send email

		s.sendmail(notSoCreativeEmail, index, msg.as_string())

		with open('log.txt','a') as f:
			f.write(f'Email sent to {index}')

	s.close()


if __name__ == '__main__':
	main()
