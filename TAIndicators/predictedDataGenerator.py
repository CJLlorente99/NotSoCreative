from numpy.random import normal

# This function simulates some simulated data. Most probably, the LSTM model could be fitted into this function somehow
# The output of the function should be a value containing the predicted value


def predictedDataGenerator(dataGetter, avgError):
    tomorrowData = dataGetter.getNextDay()

    # Simulate that we're able to predict tomorrow data perfectly but with some noise
    return tomorrowData.Close.values[0]*(1 + avgError * normal(0, 1, 1))
