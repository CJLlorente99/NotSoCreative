from numpy.random import normal


def predictedDataGenerator(dataGetter, avgError):
    tomorrowData = dataGetter.getNextDay()

    # Simulate that we're able to predict tomorrow data perfectly but with some noise
    return tomorrowData.Close.values[0]*(1 + avgError * normal(0, 1, 1))
