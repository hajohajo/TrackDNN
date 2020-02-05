from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from utilities import inputVariables
from rootFileReader import getSamples
import numpy as np
import sys

#For each variable, calculate per sample weights that transfer the training sample
#distribution to the target samples distribution. Note: the target has been chosen
#as a sample that seems to provide loads of difficult tracks that the classifier
#would not learn from the regular QCD/TTbar samples otherwise
def domainAdaptationWeights(dataframe, targetPath):
    target = getSamples([targetPath])
    quantiles = np.linspace(0.0, 1.0, 11)[1:-1]

    totalSampleWeights = np.zeros(dataframe.shape[0])
    for variable in inputVariables:
        minimum = np.min(dataframe.loc[:, variable])
        maximum = np.max(dataframe.loc[:, variable])
        input = dataframe.loc[:, variable]
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

        totalSampleWeights = totalSampleWeights+sampleWeights

    #Additionally a small factors to account for the true-fake imbalance in the dataset
    #and the imbalance between different algos
    trueFakeWeights = compute_sample_weight('balanced', dataframe.trk_isTrue)
    algoWeights = compute_sample_weight('balanced', dataframe.trk_algo)

    totalSampleWeights = totalSampleWeights + trueFakeWeights + algoWeights
    return totalSampleWeights



def splitToTrainAndTest(dataframe, frac=0.1):
    train, test = np.split(dataframe.sample(frac=1, random_state=13), [int((1.0-frac)*len(dataframe))])
    return train, test

class preprocessor():
    def __init__(self, lowerQuantile=0.1, upperQuantile=0.9):
        self.lowerQuantile = lowerQuantile
        self.upperQuantile = upperQuantile
        self.scaler = StandardScaler()
        self.fitCalled = False

    def fit(self, dataframe):
        self.fitCalled = True
        self.lowerThresholds = dataframe.loc[:, inputVariables].quantile(self.lowerQuantile)
        self.upperThresholds = dataframe.loc[:, inputVariables].quantile(self.upperQuantile)
        self.variableNamesToClip = inputVariables
        self.scaler.fit(dataframe)

    def process(self, dataframe):
        if not self.fitCalled:
            print("ERROR: tried preprocessing without having fitted preprocessor first")
            sys.exit(1)
        clipped = dataframe.loc[:, inputVariables].clip(self.lowerThresholds, self.upperThresholds, axis=1)

        clipped.loc[:, "trk_algo"] = dataframe.trk_algo
        scaled = clipped.loc[:, dataframe.columns]
        return scaled

    def getMeansAndScales(self):
        means, scales = self.scaler.mean_, self.scaler.scale_
        return means, scales
