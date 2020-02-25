from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.utils import compute_sample_weight
from utilities import inputVariables
from rootFileReader import getSamples
import numpy as np
import pandas as pd
import sys

def smoothLabels(labels):
    noise = np.random.exponential(0.3, labels.shape[0])
    index = (np.random.sample(labels.shape[0]) > 0.5)
    noise = np.subtract(noise, np.min(noise))
    noise[labels == 1] = -1.0*noise[labels == 1]
    noise[index == 1] = 0.0
    return np.clip(labels + noise, 0.0, 1.0)

def ptFakerateFlatteningWeights(dataframe):
    nEntries = dataframe.shape[0]
    # binning = np.linspace(0.0, 1000.0, 21)
    binning = np.logspace(0.0, 3, 11)

    trueIndices = dataframe.trk_isTrue == 1
    fakeIndices = dataframe.trk_isTrue == 0
    pt = np.clip(dataframe.loc[:, "trk_pt"], binning[0], binning[-1]-1e-3)
    digitizedIndices = np.digitize(pt, binning, right=True)

    #Fix the discrepancy
    digitizedIndices[digitizedIndices == 0] = 1
    digitizedIndices = digitizedIndices - 1

    binnedSignal = np.ones(len(binning))
    binnedBackground = np.ones(len(binning))

    for i in np.unique(digitizedIndices):
        binnedSignal[i] = np.sum(digitizedIndices[trueIndices] == i)
        binnedBackground[i] = np.sum(digitizedIndices[fakeIndices] == i)

    binnedTotal = np.add(binnedSignal, binnedBackground)

    binMultipliers = np.divide(nEntries/len(binning), binnedTotal)

    sigMultipliers = np.divide(binnedTotal, 2.0*binnedSignal)
    bkgMultipliers = np.divide(binnedTotal, 2.0*binnedBackground)

    sigMultipliers = np.multiply(sigMultipliers, binMultipliers)
    bkgMultipliers = np.multiply(bkgMultipliers, binMultipliers)

    sampleWeights = np.zeros(nEntries)
    sampleWeights[trueIndices] = sigMultipliers[digitizedIndices[trueIndices]]
    sampleWeights[fakeIndices] = bkgMultipliers[digitizedIndices[fakeIndices]]

    import matplotlib.pyplot as plt
    plt.hist(pt[trueIndices], binning, weights=sampleWeights[trueIndices], label="true", alpha=0.8)
    plt.hist(pt[fakeIndices], binning, weights=sampleWeights[fakeIndices], label="fake", alpha=0.8)
    plt.xscale("log")
    plt.show()

    return sampleWeights

def featureBalancingWeights(dataframe):
    transformedFrame = pd.DataFrame(PowerTransformer().fit_transform(dataframe.loc[:, inputVariables]), columns=inputVariables)
    totalSampleWeights = np.zeros(dataframe.shape[0])
    for variable in inputVariables:
        input = transformedFrame.loc[:, variable]
        minimum = np.min(input)
        maximum = np.max(input)
        binning = np.linspace(minimum, maximum, 11)
        dataframeIndexed = np.digitize(input, binning, right=True)

        weights = compute_sample_weight('balanced', dataframeIndexed)
        weights = weights/np.max(weights)

        totalSampleWeights = np.add(totalSampleWeights, weights)


    #Additionally a small factors to account for the true-fake imbalance in the dataset
    #and the imbalance between different algos
    trueFakeWeights = compute_sample_weight('balanced', dataframe.trk_isTrue)
    algoWeights = compute_sample_weight('balanced', dataframe.trk_algo)
    trueFakeWeights = trueFakeWeights/np.max(trueFakeWeights)
    algoWeights = algoWeights/np.max(algoWeights)

    totalSampleWeights = totalSampleWeights + trueFakeWeights + algoWeights
    print("min: {}, max: {}, std.dev: {}".format(np.round(np.min(totalSampleWeights), 3),
                                                 np.round(np.max(totalSampleWeights), 3),
                                                 np.round(np.std(totalSampleWeights), 3)))
    return totalSampleWeights

#For each variable, calculate per sample weights that transfer the training sample
#distribution to the target samples distribution. Note: the target has been chosen
#as a sample that seems to provide loads of difficult tracks that the classifier
#would not learn from the regular QCD/TTbar samples otherwise
def domainAdaptationWeights(dataframe, targetPath):
    target = getSamples([targetPath])
    quantiles = np.linspace(0.0, 1.0, 11)[1:-1]

    totalSampleWeights = np.zeros(dataframe.shape[0])
    for variable in inputVariables:
        input = dataframe.loc[:, variable]
        minimum = np.min(input)
        maximum = np.max(input)
        clippedTarget = np.clip(target.loc[:, variable], minimum, maximum)
        edges = np.quantile(input, quantiles)
        binning = np.insert(edges, 0, minimum)
        binning = np.append(binning, maximum)
        dataframeIndexed = np.digitize(input, binning, right=True)

        # Fix the difference between digitize and histogram binnings
        dataframeIndexed[dataframeIndexed == 0] = 1
        dataframeIndexed = dataframeIndexed - 1

        binCountsDataframe, _ = np.histogram(input, binning)
        binCountsTarget, _ = np.histogram(clippedTarget, binning)
        binCountsDataframe = binCountsDataframe/dataframe.shape[0]
        binCountsTarget = binCountsTarget/target.shape[0]

        reweightingFactorPerBin = np.zeros(len(binCountsTarget))
        for i in range(len(binCountsTarget)):
            if binCountsTarget[i] != 0:
                reweightingFactorPerBin[i] = binCountsTarget[i]/binCountsDataframe[i]
            else:
                #if the target has empty bin, do not reweight the training samples at all
                reweightingFactorPerBin[i] = 1

        sampleWeights = [reweightingFactorPerBin[i] for i in dataframeIndexed]

        totalSampleWeights = totalSampleWeights+sampleWeights/np.max(sampleWeights)

    #Additionally a small factors to account for the true-fake imbalance in the dataset
    #and the imbalance between different algos
    trueFakeWeights = compute_sample_weight('balanced', dataframe.trk_isTrue)
    algoWeights = compute_sample_weight('balanced', dataframe.trk_algo)
    trueFakeWeights = trueFakeWeights/np.max(trueFakeWeights)
    algoWeights = algoWeights/np.max(algoWeights)

    totalSampleWeights = totalSampleWeights + trueFakeWeights + algoWeights
    return totalSampleWeights



def splitToTrainAndTest(dataframe, frac=0.1):
    train, test = np.split(dataframe.sample(frac=1, random_state=13), [int((1.0-frac)*len(dataframe))])
    return train, test

class preprocessor():
    def __init__(self, lowerQuantile=0.1, upperQuantile=0.9):
        self.lowerQuantile = lowerQuantile
        self.upperQuantile = upperQuantile
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()
        self.fitCalled = False

    def fit(self, dataframe):
        self.fitCalled = True
        self.lowerThresholds = dataframe.loc[:, inputVariables].quantile(self.lowerQuantile)
        self.upperThresholds = dataframe.loc[:, inputVariables].quantile(self.upperQuantile)
        self.variableNamesToClip = inputVariables

        clipped = dataframe.loc[:, inputVariables].clip(self.lowerThresholds, self.upperThresholds, axis=1)
        clipped.loc[:, "trk_algo"] = dataframe.trk_algo
        self.scaler.fit(clipped)

    def process(self, dataframe):
        if not self.fitCalled:
            print("ERROR: tried preprocessing without having fitted preprocessor first")
            sys.exit(1)
        clipped = dataframe.loc[:, inputVariables].clip(self.lowerThresholds, self.upperThresholds, axis=1)

        clipped.loc[:, "trk_algo"] = dataframe.trk_algo
        # scaled = clipped.loc[:, dataframe.columns]
        scaled = pd.DataFrame(self.scaler.transform(clipped), columns=dataframe.columns.values)
        # return scaled
        return dataframe

    def getMeansAndScales(self):
        means, scales = self.scaler.mean_, self.scaler.scale_
        return means, scales
