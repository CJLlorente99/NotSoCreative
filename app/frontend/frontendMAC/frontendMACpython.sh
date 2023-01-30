python -m ensurepip > /dev/null 2>&1

python ./installPackage.py customtkinter > /dev/null 2>&1
python ./installPackage.py pandas > /dev/null 2>&1
python ./installPackage.py matplotlib > /dev/null 2>&1
python ./installPackage.py tk > /dev/null 2>&1
python ./installPackage.py mplfinance > /dev/null 2>&1
python ./installPackage.py Pillow > /dev/null 2>&1
python ./installPackage.py yfinance > /dev/null 2>&1
python ./installPackage.py numpy > /dev/null 2>&1
python ./installPackage.py google-cloud-storage > /dev/null 2>&1

python ./frontendMAC.py > /dev/null 2>&1
exit
