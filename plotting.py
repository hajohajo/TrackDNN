from rootFileReader import getSamples
from utilities import inputVariables, Algo
import numpy as np
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def createClassifierPlots(classifier, preproc):
    samplePathToName = {'datasets/T1tttt.root': 'T1tttt', 'datasets/T5qqqqWW.root': 'T5qqqqWW', 'datasets/QCD_flat_small.root': 'QCD'}
    listOfSamplePaths = ['datasets/T1tttt.root', 'datasets/T5qqqqWW.root', 'datasets/QCD_flat_small.root']
    for samplePath in listOfSamplePaths:
        name = samplePathToName[samplePath]
        sample = getSamples([samplePath])
        samplePreprocessed = preproc.process(sample.loc[:, inputVariables+["trk_algo"]])
        sample.loc[:, "trk_dnn"] = 2*classifier.predict(samplePreprocessed)-1.0

        createBinnedMeanPlot(sample, 'trk_pt', 'trk_dnn', np.logspace(-1, 3), name+"MVAVsPt")
        createROCplot(sample, name)
        createMVADistribution(sample, name)

def createROCplot(dataframe, name):
    for step in np.append(np.unique(dataframe.trk_algo), 99):
        indexConditions = (dataframe.trk_algo==step)
        if(step==99):
            indexConditions=np.ones(dataframe.shape[0], dtype=bool)
        falsePositiveRate, truePositiveRate, thresholds = roc_curve(dataframe.loc[indexConditions, "trk_isTrue"], dataframe.loc[indexConditions, "trk_dnn"])
        falsePositiveRateBase, truePositiveRateBase, thresholdsBase = roc_curve(dataframe.loc[indexConditions, "trk_isTrue"], dataframe.loc[indexConditions, "trk_mva"])

        aucDNN = auc(falsePositiveRate, truePositiveRate)
        aucBase = auc(falsePositiveRateBase, truePositiveRateBase)

        plt.plot(truePositiveRateBase, 1-falsePositiveRateBase, label="BDT, area = {:.3f}".format(aucBase))
        plt.plot(truePositiveRate, 1-falsePositiveRate, label="DNN, area = {:.3f}".format(aucDNN))
        plt.legend()
        plt.title("ROC curves, "+Algo.toString(step))
        plt.ylabel("Fake track rejection")
        plt.xlabel("True track efficiency")
        plt.savefig("./plots/"+name+"_"+Algo.toString(step)+"_ROC.pdf")
        plt.clf()

def createBinnedMeanPlot(dataframe, xvariable, yvariable, binning, name):

    digitizedIndices = np.digitize(np.clip(dataframe.loc[:, xvariable], binning[0], binning[-1] - 1e-6), binning) - 1

    for isTrue in [0, 1]:
        truthLabel = "True"
        if isTrue==0:
            truthLabel = "Fake"
        for step in np.unique(dataframe.trk_algo):
            means = np.zeros(len(binning))
            stds = np.zeros(len(binning))
            meansBase = np.zeros(len(binning))
            stdsBase = np.zeros(len(binning))

            for i in range(len(binning)):
                entriesMask = (digitizedIndices==i) & (dataframe.trk_algo==step) & (dataframe.trk_isTrue==isTrue)
                if(np.sum(entriesMask)>1):
                    means[i] = np.mean(dataframe.loc[entriesMask, yvariable])
                    stds[i] = np.std(dataframe.loc[entriesMask, yvariable])
                    meansBase[i] = np.mean(dataframe.loc[entriesMask, "trk_mva"])
                    stdsBase[i] = np.std(dataframe.loc[entriesMask, "trk_mva"])

            binCenters = binning[:] + (binning[1]-binning[0])/2.0

            plt.errorbar(binCenters, meansBase, yerr=stdsBase, fmt='*', label="BDT")
            plt.errorbar(binCenters, means, yerr=stds, fmt='*', label="DNN")
            plt.legend()
            plt.title(name+" "+Algo.toString(step)+", "+truthLabel+" tracks")
            plt.xlabel(xvariable)
            plt.ylabel(yvariable)
            plt.ylim(-1.2, 1.2)
            plt.xscale('log')
            plt.savefig("./plots/"+name+"_"+truthLabel+"_"+Algo.toString(step)+".pdf")
            plt.clf()

def createMVADistribution(dataframe, name):
    binning = np.linspace(-1.0, 1.0, 41)
    for step in np.unique(dataframe.trk_algo):
        trueTracks = (dataframe.trk_algo == step) & (dataframe.trk_isTrue == 1)
        fakeTracks = (dataframe.trk_algo == step) & (dataframe.trk_isTrue == 0)

        plt.hist(dataframe.loc[trueTracks, "trk_dnn"], bins=binning, label="True", density=True, alpha=0.7, edgecolor='black', linewidth=1.0)
        plt.hist(dataframe.loc[fakeTracks, "trk_dnn"], bins=binning, label="Fake", density=True, alpha=0.7, edgecolor='black', linewidth=1.0)

        plt.legend()
        plt.xlabel("MVA output")
        plt.ylabel("Normalized tracks")
        plt.title("MVA distribution")

        plt.savefig("./plots/"+name+"_"+Algo.toString(step)+"_MVADistribution.pdf")
        plt.clf()

