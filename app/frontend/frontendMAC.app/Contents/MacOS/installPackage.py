import pip
import sys

def import_or_install(package):
	try:
		__import__(package)
	except ImportError:
		pip.main(['install', package])

if __name__ == '__main__':
	import_or_install(sys.argv[1])