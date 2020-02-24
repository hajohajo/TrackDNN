#The minimal interface to read root ntuples into something more friendly to use
#I hate ROOT and you should too.
from root_pandas import read_root

def getSamples(listOfNames):
    columnsToRead = ['trk_isTrue', 'trk_mva', 'trk_pt', 'trk_eta', 'trk_dxy', 'trk_dz', 'trk_dxyClosestPV', 'trk_dzClosestPV', 'trk_ptErr',
                     'trk_etaErr', 'trk_dxyErr', 'trk_dzErr', 'trk_nChi2', 'trk_ndof', 'trk_nLost', 'trk_nPixel', 'trk_nStrip',
                     'trk_algo']

    #test if the root file is a flattened nTuple or not. Not a very pretty test
    t_ = read_root(listOfNames[0], columns=["trk_isTrue"], chunksize=1).__iter__().__next__()
    notFlat = hasattr(t_.loc[t_.index.values[0], "trk_isTrue"], "__len__")
    if notFlat:
        dataframe = read_root(listOfNames, columns=columnsToRead, flatten=columnsToRead)
        dataframe.drop("__array_index", axis=1, inplace=True)
    else:
        dataframe = read_root(listOfNames, columns=columnsToRead)

    return dataframe