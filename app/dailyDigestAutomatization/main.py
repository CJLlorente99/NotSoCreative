import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from renderer import Renderer
import pytz
from datetime import datetime

"""
This file is to be called after the stock market is closed in order to send the daily digest
1) Perform checks (has stock market close already, has stock market opened today)
2) Instantiate a Renderer and render the two types of html strings (long and short)
3) Load email information (emails and type of digest)
4) Send emails
"""

notSoCreativeEmail = 'notsocreative2023@gmail.com'
destinationEmail = 'paulleonardo.heller@stud.tu-darmstadt.de'

def main():

	"""
	1) Perform checks (has stock market close already, has stock market opened today)
	"""

	"""
	2) Instantiate a Renderer and render the two types of html strings (long and short)
	"""
	todayDate = datetime.now(pytz.timezone('America/New_York'))
	rend = Renderer(todayDate)
	renderedShortHTML = rend.renderShortDailyDigest()
	renderedLongHTML = rend.renderLongDailyDigest()

	"""
	3) Load email information (emails and type of digest)
	"""

	"""
	4) Send emails
	"""
	# Create a msg with the different parts
	msg = MIMEMultipart('alternative')
	msg['Subject'] = '[DataScienceII] Daily Digest ' + todayDate.strftime('%d/%m/%Y')
	msg['From'] = notSoCreativeEmail
	msg['To'] = destinationEmail
	msg.attach(MIMEText(renderedShortHTML, 'html'))

	# Connect with SMTP gmail server and send the email through it
	s = smtplib.SMTP('smtp.gmail.com', 587)
	s.ehlo()
	s.starttls()
	s.ehlo()
	s.login(notSoCreativeEmail, 'nfkpbmnkifameswu')
	s.sendmail(notSoCreativeEmail, destinationEmail, msg.as_string())
	s.close()


if __name__ == '__main__':
	main()
