from inversionStrategyJSONAPI import *


jsonFile = './strategies.json'

# json manager
jsonmanager = JsonStrategyManager(jsonFile)

# Delete all
jsonmanager.deleteFile()

# BIA
biaStrategy = Strategy('bia_24_1_2022', 'Best Invesment Approach for benchmark purposes', [])
jsonmanager.addStrategy(biaStrategy)

# WIA
wiaStrategy = Strategy('wia_24_1_2022', 'Worst Invesment Approach for benchmark purposes', [])
jsonmanager.addStrategy(wiaStrategy)

# CA
caStrategy = Strategy('ca_24_1_2022', 'Continuous Average strategy for benchmark purposes', [])
jsonmanager.addStrategy(caStrategy)

# Idle
idleStrategy = Strategy('idle_24_1_2022', 'Do Nothing strategy for benchmark purposes', [])
jsonmanager.addStrategy(idleStrategy)

# Random
randomStrat = Strategy('random_24_1_2022', 'Random strategy for benchmark purposes', [])
jsonmanager.addStrategy(randomStrat)

# BaH
bahStrategy = Strategy('bah_24_1_2022', 'Buy and Hold strategy for benchmark purposes', [])
jsonmanager.addStrategy(bahStrategy)

# BIA
biaStrategy = Strategy('bia_25_1_2022', 'Best Invesment Approach for benchmark purposes', [])
jsonmanager.addStrategy(biaStrategy)

# WIA
wiaStrategy = Strategy('wia_25_1_2022', 'Worst Invesment Approach for benchmark purposes', [])
jsonmanager.addStrategy(wiaStrategy)

# CA
caStrategy = Strategy('ca_25_1_2022', 'Continuous Average strategy for benchmark purposes', [])
jsonmanager.addStrategy(caStrategy)

# Idle
idleStrategy = Strategy('idle_25_1_2022', 'Do Nothing strategy for benchmark purposes', [])
jsonmanager.addStrategy(idleStrategy)

# Random
randomStrat = Strategy('random_25_1_2022', 'Random strategy for benchmark purposes', [])
jsonmanager.addStrategy(randomStrat)

# BaH
bahStrategy = Strategy('bah_25_1_2022', 'Buy and Hold strategy for benchmark purposes', [])
jsonmanager.addStrategy(bahStrategy)

# BiLSTMWindowRobMMT1T2Legacy
inputs = []
# Return_outra
inputs.append(StrategyInput('Return_interday', '', 'Return_outra', 'Log', []))
# Volume
inputs.append(StrategyInput('Volume', '', 'Volume', '', []))
# adx_pos_w50
param1 = InputParameter('Window', 50)
inputs.append(StrategyInput('adx', '', 'adx_pos_w50', 'adx_pos', [param1]))
# bb_lband_w31_stdDev3.5483563609690094
param1 = InputParameter('Window', 31)
param2 = InputParameter('StdDev', 3.5483563609690094)
inputs.append(StrategyInput('bb', '', 'bb_lband_w31_stdDev3.5483563609690094', 'lband', [param1, param2]))
# Diff_open
inputs.append(StrategyInput('Diff_open', '', 'Diff_open', '', []))
# rsi_w44
param1 = InputParameter('Window', 44)
inputs.append(StrategyInput('rsi', '', 'rsi_w44', 'rsi', [param1]))
# adx_neg_w14
param1 = InputParameter('Window', 14)
inputs.append(StrategyInput('adx', '', 'adx_neg_w14', 'adx_neg', [param1]))
# stochRsi_d_w4_s131_s27
param1 = InputParameter('Window', 4)
param2 = InputParameter('Smooth1', 31)
param3 = InputParameter('Smooth2', 7)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w4_s131_s27', 'd', [param1, param2, param3]))
# adx_w50
param1 = InputParameter('Window', 50)
inputs.append(StrategyInput('adx', '', 'adx_w50', 'adx', [param1]))
# Return_intra
inputs.append(StrategyInput('Return_intraday', '', 'Return_intraday', 'Log', []))
# macd_difffW12_sW44_signal3
param1 = InputParameter('FastWindow', 12)
param2 = InputParameter('SlowWindow', 44)
param3 = InputParameter('Signal', 3)
inputs.append(StrategyInput('macd', '', 'macd_difffW12_sW44_signal3', 'diff', [param1, param2, param3]))
# stochRsi_d_w50_s110_s218
param1 = InputParameter('Window', 50)
param2 = InputParameter('Smooth1', 10)
param3 = InputParameter('Smooth2', 18)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w50_s110_s218', 'd', [param1, param2, param3]))
# stochRsi_d_w3_s132_s247
param1 = InputParameter('Window', 3)
param2 = InputParameter('Smooth1', 32)
param3 = InputParameter('Smooth2', 47)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w3_s132_s247', 'd', [param1, param2, param3]))
# macd_fW1_sW41_signal34
param1 = InputParameter('FastWindow', 1)
param2 = InputParameter('SlowWindow', 41)
param3 = InputParameter('Signal', 34)
inputs.append(StrategyInput('macd', '', 'macd_fW1_sW41_signal34', 'macd', [param1, param2, param3]))
bilstmWindowRobMMT1T2LegacyStrategy = Strategy('bilstmWindowRobMMT1T2Legacy_24_1_2023', '', inputs)
jsonmanager.addStrategy(bilstmWindowRobMMT1T2LegacyStrategy)

# BiLSTMWindowRobMMT1T2Legacy
inputs = []
# Return_outra
inputs.append(StrategyInput('Return_interday', '', 'Return_outra', 'Log', []))
# Volume
inputs.append(StrategyInput('Volume', '', 'Volume', '', []))
# adx_pos_w50
param1 = InputParameter('Window', 50)
inputs.append(StrategyInput('adx', '', 'adx_pos_w50', 'adx_pos', [param1]))
# bb_lband_w31_stdDev3.5483563609690094
param1 = InputParameter('Window', 31)
param2 = InputParameter('StdDev', 3.5483563609690094)
inputs.append(StrategyInput('bb', '', 'bb_lband_w31_stdDev3.5483563609690094', 'lband', [param1, param2]))
# Diff_open
inputs.append(StrategyInput('Diff_open', '', 'Diff_open', '', []))
# rsi_w44
param1 = InputParameter('Window', 44)
inputs.append(StrategyInput('rsi', '', 'rsi_w44', 'rsi', [param1]))
# adx_neg_w14
param1 = InputParameter('Window', 14)
inputs.append(StrategyInput('adx', '', 'adx_neg_w14', 'adx_neg', [param1]))
# stochRsi_d_w4_s131_s27
param1 = InputParameter('Window', 4)
param2 = InputParameter('Smooth1', 31)
param3 = InputParameter('Smooth2', 7)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w4_s131_s27', 'd', [param1, param2, param3]))
# adx_w50
param1 = InputParameter('Window', 50)
inputs.append(StrategyInput('adx', '', 'adx_w50', 'adx', [param1]))
# Return_intra
inputs.append(StrategyInput('Return_intraday', '', 'Return_intraday', 'Log', []))
# macd_difffW12_sW44_signal3
param1 = InputParameter('FastWindow', 12)
param2 = InputParameter('SlowWindow', 44)
param3 = InputParameter('Signal', 3)
inputs.append(StrategyInput('macd', '', 'macd_difffW12_sW44_signal3', 'diff', [param1, param2, param3]))
# stochRsi_d_w50_s110_s218
param1 = InputParameter('Window', 50)
param2 = InputParameter('Smooth1', 10)
param3 = InputParameter('Smooth2', 18)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w50_s110_s218', 'd', [param1, param2, param3]))
# stochRsi_d_w3_s132_s247
param1 = InputParameter('Window', 3)
param2 = InputParameter('Smooth1', 32)
param3 = InputParameter('Smooth2', 47)
inputs.append(StrategyInput('stochasticRsi', '', 'stochRsi_d_w3_s132_s247', 'd', [param1, param2, param3]))
# macd_fW1_sW41_signal34
param1 = InputParameter('FastWindow', 1)
param2 = InputParameter('SlowWindow', 41)
param3 = InputParameter('Signal', 34)
inputs.append(StrategyInput('macd', '', 'macd_fW1_sW41_signal34', 'macd', [param1, param2, param3]))
bilstmWindowRobMMT1T2LegacyStrategy = Strategy('bilstmWindowRobMMT1T2Legacy_25_1_2023', '', inputs)
jsonmanager.addStrategy(bilstmWindowRobMMT1T2LegacyStrategy)