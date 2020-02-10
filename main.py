from rootFileReader import getSamples
import pandas as pd
import tensorflow as tf
import numpy as np
from utilities import createFolders, inputVariables
from neuralNetworks import createClassifier, createFrozenModel
from preprocessing import preprocessor, domainAdaptationWeights
from plotting import createClassifierPlots
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', None)

tf.compat.v2.enable_v2_behavior()

import tensorflow_addons as tfa

def main():
    # adam = tfa.optimizers.RectifiedAdam(lr=1e-3,
    #     total_steps=10000,
    #     warmup_proportion=0.1,
    #     min_lr=1e-5)

    adam = tfa.optimizers.LAMB()

    print("Begin program")


    createFolders(["plots"])

    path = "~/QCD_Flat_15_7000_correct/"
    QCDTrain = getSamples([path+"trackingNtuple.root", path+"trackingNtuple2.root", path+"trackingNtuple3.root", path+"trackingNtuple4.root"])
    # QCDTrain = QCDTrain.sample(n=100000)
    weights = domainAdaptationWeights(QCDTrain, "datasets/T5qqqqWW.root")
    preproc = preprocessor(0.05, 0.95)
    preproc.fit(QCDTrain.loc[:, inputVariables+["trk_algo"]])

    # QCDTrainPreprocessed = preproc.process(QCDTrain.loc[:, inputVariables+["trk_algo"]])
    QCDTrainPreprocessed = QCDTrain.loc[:, inputVariables+["trk_algo"]]

    means, scales = preproc.getMeansAndScales()

    #The outputs of these printouts are to be used as the cutoff values when evaluating the
    #deployed model in CMSSW. See RecoTracker/FinalTrackSelectors/plugins/TrackTFClassifier.cc
    print(preproc.variableNamesToClip)
    print("Upper cutoffs: ", np.round(preproc.upperThresholds.to_numpy(), 3).tolist())
    print("Lower cutoffs: ", np.round(preproc.lowerThresholds.to_numpy(), 3).tolist())

    classifier = createClassifier(len(QCDTrainPreprocessed.columns), means, scales)
    # classifier.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, amsgrad=True),
    classifier.compile(optimizer=adam,
                       metrics=[tf.keras.metrics.AUC(name="auc")],
                       # loss="mse")
                       loss = "binary_crossentropy")

    print(weights)
    classifier.fit(QCDTrainPreprocessed.to_numpy(),
                   QCDTrain.loc[:, "trk_isTrue"],
                   sample_weight=weights,
                   epochs=20,
                   batch_size=16384,
                   # validation_split=0.5)
                   validation_split = 0.1)

    #Saving model in case later need for additional plotting arises
    classifier.save('./model.h5')

    #Create quick set of plots to get an idea of the performance
    #The true plots have to be done in CMSSW
    createClassifierPlots(classifier, preproc)

    #Model for deployment using CMSSW TF C++API
    createFrozenModel(classifier)

if __name__ == "__main__":
    main()