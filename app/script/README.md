# Format of CSV generated #
This file will be saved in **scriptData/myData.csv** and **can** be read by the frontend SW. The format
of the CSV will be the one depicted in the table below.

| Date                | investorStrategt | MoneyInvested | MoneyNotInvested | MoneyInvestedToday | PerInvestToday | TotalPortfolioValue | Indicators & Data |
|---------------------|------------------|---------------|------------------|--------------------|----------------|---------------------|-------------------|
| YYYY-MM-DD HH:MM:SS | String           | Float         | Float            | Float              | Float          | Float               | Float             |

This CSV will also store the information for the benchmarks and all other strategies that could be tried.
Accepted values for investorStrategy are:

*(Benchmark)*
- bia
- wia
- bah
- idle
- random
- ca

*(OtherStrategies)*
- rsi
- bb

# Format of JSON (strategies definition) #

A JSON file contains the description of all the strategies that are calculated on each run.

Make use of the inversionStrategyJSONAPY.py classes to add new strategies with their proper
inputs and their proper parameters. Example of JSON file describing one strategy.

```
[
	{
		"Name": "StrategyName",
		"Description": "StrategyDescription",
		"Inputs": [
			{
				"Name": "macd",
				"DfName": "macd_fW10_sW26_s15",
				"Key": "macd",
				"Description": "MACD indicator",
				"Parameters": [
					{
						"Name": "FastWindow",
						"Value": 10
					},
					{
						"Name": "SlowWindow",
						"Value": 26
					},
					{
						"Name": "Signal",
						"Value": 15
					}
				]
			},
			{
				"Name": "Low",
				"DfName": "Low",
				"Key": "none",
				"Description": "Low value from SP500",
				"Parameters": []
			}
		]
	},
	{
		"Name": "bia",
		"Description": "BIA benchmark",
		"Inputs": [
			{
				"Name": "Open",
				"DfName": "Open",
				"Key": "Log",
				"Description": "Open log value from SP500",
				"Parameters": []
			}
		]
	}
]
````

All attributes named 'Name' are hard-coded and therefore can only take specific values.
The following table shows the input names, parameters names and key names accepted.

| Input Name    | Key Names                             | Parameter Names                |
|---------------|---------------------------------------|--------------------------------|
| High          | Natural                               |                                |
| Low           | Natural                               |                                |
| Volume        | Natural                               |                                |
| Close         | Natural, Log                          |                                |
| Open          | Natural, Log                          |                                |
| adi           | acc_dist_index                        |                                |
| adx           | adx, adx_neg, adx_pos                 | Window                         |
| aroon         | aroon_indicator, aroon_down, aroon_up | Window                         |
| atr           | average_true_range                    | Window                         |
| bb            | pband, mavg, hband, lband             | Window, StdDev                 |
| ema           | ema                                   | Window                         |
| macd          | macd, signal, diff                    | FastWindow, SlowWindow, Signal |
| obv           | on_balance_volume                     |                                |
| rsi           | rsi                                   | Window                         |
| stochasticRsi | stochrsi, k, d                        | Window, Smooth1, Smooth2       |

# How to run it automatically #
## Windows ##
1. Verify all needed libraries are installed. To do this, run "run_script.bat" from the terminal. This is done
by just opening a PowerShell terminal and writing 
``./run_script.bat``. Install the libraries with ``pip install *``
2. Open the Windows "Task Scheduler".
3. Create two basic tasks by clicking on "Create basic task". Give name and description as you want. Just select a daily
periodicity at 16.00 (CET) for one task and 22.30 (CET) for the other.

## Mac ##
1. Move files "local.ds2Afternoon.plist" and "local.ds2Morning.plist" into "/Library/LaunchDaemons"
2. Load the scripts by making use of "launchctl". In the terminal

````
launchctl load ~/Library/LaunchAgents/local.ds2Afternoon.plist 
launchctl load ~/Library/LaunchAgents/local.ds2Morning.plist
````

Once this is done, it will be done forever until unloaded by writing the following in the terminal.
````
launchctl unload ~/Library/LaunchAgents/local.ds2Afternoon.plist
launchctl unload ~/Library/LaunchAgents/local.ds2Morning.plist
````