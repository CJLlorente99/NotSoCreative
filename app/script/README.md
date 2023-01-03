# Format of CSV generated #
This file will be saved in **scriptData/myData.csv** and **can** be read by the frontend SW. The format
of the CSV will be the one depicted in the table below.

| Date       | investorStrategy | MoneyInvested | MoneyNotInvested | MoneyBoughtToday | MoneySoldToday | PerBoughtTomorrow | PerSoldTomorrow | TotalPortfolioValue | TodayOpen | YesterdayClose |
|------------|------------------|---------------|------------------|------------------|----------------|-------------------|-----------------|---------------------|-----------|----------------|
| YYYY-MM-DD |      String      |     Float     |       Float      |       Float      |      Float     | Float             | Float           |        Float        |   Float   |      Float     |

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
- 

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

