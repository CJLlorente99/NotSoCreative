python -m ensurepip

python ./installPackage.py customtkinter
python ./installPackage.py pandas
python ./installPackage.py matplotlib
python ./installPackage.py tk
python ./installPackage.py mplfinance
python ./installPackage.py Pillow
python ./installPackage.py yfinance
python ./installPackage.py numpy
python ./installPackage.py google-cloud-storage

python ./frontendMAC.py > /dev/null 2>&1
exit
