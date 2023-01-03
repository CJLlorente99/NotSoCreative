# Format of CSV generated #
This file will be saved in **scriptData/myData.csv** and **can** be read by the frontend SW. The format
of the CSV will be the one depicted in the table below.

| Date       | investorStrategy | MoneyInvested | MoneyNotInvested | MoneyBoughtToday | MoneySoldToday | MoneyBoughtTomorrow | MoneySoldTomorrow | TotalPortfolioValue | TodayOpen | YesterdayClose |
|------------|------------------|---------------|------------------|------------------|----------------|---------------------|-------------------|---------------------|-----------|----------------|
| YYYY-MM-DD |      String      |     Float     |       Float      |       Float      |      Float     |        Float        |       Float       |        Float        |   Float   |      Float     |

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