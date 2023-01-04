from inversionStrategyJSONAPI import *


jsonFile = 'ACHTUNGScriptData/strategies.json'

# json manager
jsonmanager = JsonStrategyManager(jsonFile)

# Delete all
jsonmanager.deleteFile()

# CA
caStrategy = Strategy('ca', 'Continuous Average strategy for benchmark purposes', [])
jsonmanager.addStrategy(caStrategy)

# Idle
idleStrategy = Strategy('idle', 'Do Nothing strategy for benchmark purposes', [])
jsonmanager.addStrategy(idleStrategy)

# Random
randomStrat = Strategy('random', 'Random strategy for benchmark purposes', [])
jsonmanager.addStrategy(randomStrat)

# BaH
bahStrategy = Strategy('bah', 'Buy and Hold strategy for benchmark purposes', [])
jsonmanager.addStrategy(bahStrategy)

# Example RSI
param1 = InputParameter('Window', 3)
inp1 = StrategyInput('rsi', 'RSI value', 'rsi_w3', 'rsi', [param1])
rsiStrategy = Strategy('rsi', 'Strategy based on RSI', [inp1])
jsonmanager.addStrategy(rsiStrategy)

# Example BB
param1 = InputParameter('Window', 10)
param2 = InputParameter('StdDev', 1.5)
inp1 = StrategyInput('bb', 'BB value', 'bb_w10_stdDev1.5', 'pband', [param1, param2])
bbStrategy = Strategy('bb', 'Strategy based on BB', [inp1])
jsonmanager.addStrategy(bbStrategy)