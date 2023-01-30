python3 -m ensurepip > /dev/null 2>&1

python3 ./installPackage.py customtkinter > /dev/null 2>&1
python3 ./installPackage.py pandas > /dev/null 2>&1
python3 ./installPackage.py matplotlib > /dev/null 2>&1
python3 ./installPackage.py tk > /dev/null 2>&1
python3 ./installPackage.py mplfinance > /dev/null 2>&1
python3 ./installPackage.py Pillow > /dev/null 2>&1
python3 ./installPackage.py yfinance > /dev/null 2>&1
python3 ./installPackage.py numpy > /dev/null 2>&1
python3 ./installPackage.py google-cloud-storage > /dev/null 2>&1

python3 ./frontendMAC.py > /dev/null 2>&1
exit
