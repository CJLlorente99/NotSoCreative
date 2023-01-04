from inversionStrategyJSONAPI import *


jsonFile = 'ACHTUNGScriptData/strategies.json'

# json manager
jsonmanager = JsonStrategyManager(jsonFile)

# Delete all
jsonmanager.deleteFile()

# BIA
biaStrategy = Strategy('bia', 'Best Invesment Approach for benchmark purposes', [])
jsonmanager.addStrategy(biaStrategy)

# WIA
wiaStrategy = Strategy('wia', 'Worst Invesment Approach for benchmark purposes', [])
jsonmanager.addStrategy(wiaStrategy)

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

# LSTMConfidenceOpenClose
inputs = []
# Return_interday
inputs.append(StrategyInput('Return_interday', '', 'Return_interday', 'Log', []))
# bb_pband_w3_stdDev1.774447792366109
param1 = InputParameter('Window', 3)
param2 = InputParameter('StdDev', 1.774)
inputs.append(StrategyInput('bb', 'BB value', 'bb_pband_w3_stdDev1.774447792366109', 'pband', [param1, param2]))
# Return_open
inputs.append(StrategyInput('Return_open', '', 'Return_open', 'Log', []))
# adx_pos_w6
param1 = InputParameter('Window', 6)
inputs.append(StrategyInput('adx', '', 'adx_pos_w6', 'adx_pos', [param1]))
# adx_pos_w42
param1 = InputParameter('Window', 42)
inputs.append(StrategyInput('adx', '', 'adx_pos_w6', 'adx_pos', [param1]))
# Volume
inputs.append(StrategyInput('Volume', '', 'Volume', 'Natural', []))
# adx_neg_w1
param1 = InputParameter('Window', 1)
inputs.append(StrategyInput('adx', '', 'adx_neg_w1', 'adx_neg', [param1]))
# Return_intraday
inputs.append(StrategyInput('Return_intraday', '', 'Return_intraday', 'Log', []))
# stochRsi_k_w47_s143_s212
param1 = InputParameter('Window', 47)
param2 = InputParameter('Smooth1', 43)
param3 = InputParameter('Smooth2', 12)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_k_w47_s143_s212', 'k', [param1, param2, param3]))
# stochRsi_d_w9_s144_s246
param1 = InputParameter('Window', 9)
param2 = InputParameter('Smooth1', 44)
param3 = InputParameter('Smooth2', 46)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w9_s144_s246', 'd', [param1, param2, param3]))
# stochRsi_d_w4_s16_s233
param1 = InputParameter('Window', 4)
param2 = InputParameter('Smooth1', 6)
param3 = InputParameter('Smooth2', 33)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w4_s16_s233', 'd', [param1, param2, param3]))
# adx_w10
param1 = InputParameter('Window', 10)
inputs.append(StrategyInput('adx', '', 'adx_w10', 'adx', [param1]))
# bb_pband_w7_stdDev1.4065306043590475
param1 = InputParameter('Window', 7)
param2 = InputParameter('StdDev', 1.407)
inputs.append(StrategyInput('bb', 'BB value', 'bb_pband_w7_stdDev1.4065306043590475', 'pband', [param1, param2]))
# bb_pband_w13_stdDev1.7961852973078898
param1 = InputParameter('Window', 13)
param2 = InputParameter('StdDev', 1.796)
inputs.append(StrategyInput('bb', 'BB value', 'bb_pband_w13_stdDev1.7961852973078898', 'pband', [param1, param2]))
# adx_w18
param1 = InputParameter('Window', 18)
inputs.append(StrategyInput('adx', '', 'adx_w18', 'adx', [param1]))
# stochRsi_k_w4_s16_s233
param1 = InputParameter('Window', 4)
param2 = InputParameter('Smooth1', 6)
param3 = InputParameter('Smooth2', 33)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w4_s16_s233', 'k', [param1, param2, param3]))
# adx_neg_w25
param1 = InputParameter('Window', 25)
inputs.append(StrategyInput('adx', '', 'adx_neg_w25', 'adx_neg', [param1]))
# stochRsi_d_w12_s125_s25
param1 = InputParameter('Window', 12)
param2 = InputParameter('Smooth1', 25)
param3 = InputParameter('Smooth2', 5)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w12_s125_s25', 'd', [param1, param2, param3]))
# macd_difffW5_sW39_signal14
param1 = InputParameter('FastWindow', 5)
param2 = InputParameter('SlowWindow', 39)
param3 = InputParameter('Signal', 14)
inputs.append(StrategyInput('macd', '', 'macd_difffW5_sW39_signal14', 'diff', [param1, param2, param3]))
# stochRsi_k_w29_s18_s219
param1 = InputParameter('Window', 29)
param2 = InputParameter('Smooth1', 8)
param3 = InputParameter('Smooth2', 19)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_k_w29_s18_s219', 'k', [param1, param2, param3]))

# lstmConfidence = Strategy('lstmConfidenceOpenClose', 'Strategy based on LSTM prediction of intraday return', inputs)
# jsonmanager.addStrategy(lstmConfidence)