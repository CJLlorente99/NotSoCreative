python3 -m ensurepip

python3 ./installPackage.py customtkinter
python3 ./installPackage.py pandas
python3 ./installPackage.py matplotlib
python3 ./installPackage.py tk
python3 ./installPackage.py mplfinance
python3 ./installPackage.py Pillow
python3 ./installPackage.py yfinance
python3 ./installPackage.py numpy
python3 ./installPackage.py google-cloud-storage

python3 ./frontendMAC.py > /dev/null 2>&1
exit
