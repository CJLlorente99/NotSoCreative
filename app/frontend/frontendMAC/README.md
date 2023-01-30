# MAC Frontend Instructions

This documents informs about the steps that need to be followed in order to execute the frontend in a MACOS device.

1. Install last [Python distribution](https://www.python.org/downloads/).
2. Check installation by writing the following command in the terminal.
````commandline
python3 --version
````
3. Open a terminal in the same folder where _frontendMAC.py_, _frontendMACpython.sh_, 
_googleStorageAPI_, _installPackage.py_, _png.png_ and _application_default_credentials.json_. Write the following command in the terminal.
````commandline
sh ./frontendMACInstallationpython3.sh
````

4. Frontend application should have started. If this is not the case, please contact back to the NotSoCreative Team.
5. It is only necessary to follow this procedure once. For the next time the app is to be opened, write the following in
the terminal.
````commandline
sh ./frontendMACpython3.sh
````

# Trobleshooting
If you already had a Python distribution installed in your computer and you have not installed
the most recent one, it is possible that the installation fails.

If this is the case, check you indeed have a Python distribution via writing the following command in the console.

````commandline
python --version
````

If you get a result with the version number. Then you can try the following command, after you have
positioned yourself in the folder described in the step **3**.

````commandline
sh ./frontendMACInstallationpython.sh
````

If this command results positively, next time it will only be necessary to execute the following command.
````commandline
sh ./frontendMACpython.sh
````

It is important to note that executing the frontend with Python old distributions can result
in further errors. Consider update your actual Python distribution.