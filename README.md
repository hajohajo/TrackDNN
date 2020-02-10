# TrackDNN

Hopefully a fairly short and clear framework for training track classifiers without too many complications.
It mainly relies on pandas dataframes for passing around the data and uses TensorFlow2 and its accompanying
libraries to perform the training. root_pandas package is responsible for the conversion from .root to pandas
formats.

## Environment
The requirements.txt file specifies what my python environment has when running this on my laptop.

## Inputs
The only thing not included in the repository is the training data needed. For quick testing to understand how the framework
is running, one can try the smaller datasets in the datasets/ folder. However their main purpose is to produce a bunch of plots
to get an idea of the network behaviour without the need for a full deployment and validation run in the CMSSW.

To generate inputs one should look into the TrackingNtuple (https://github.com/cms-sw/cmssw/tree/master/Validation/RecoTrack)
Remember to configure it from trackingNtuple_cff to include all you need, but not too much more to keep the training data
in reasonable size.

## Outputs
The training will by default save the model both as a .h5 file and a frozen graph .pb file. It will also produce a bunch of
plots to give an idea of the performance.

## Network
One can fiddle around the network or create your own in the neuralNetworks.py. Also all neuralNetwork related helper functions
like a function to create the frozen graph for deployment are there.

There is an example of a custom layer, StandardScalerLayer applying the scaling operation on all inputs coming to the network,
and an example of a custom activation function swish there as well.
