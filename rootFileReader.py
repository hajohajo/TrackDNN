#The minimal interface to read root ntuples into something more friendly to use
#I hate ROOT and you should too.
from root_pandas import read_root

def getSamples(listOfNames):
    columnsToRead = ['trk_isTrue', 'trk_mva', 'trk_pt', 'trk_eta', 'trk_dxy', 'trk_dz', 'trk_dxyClosestPV', 'trk_dzClosestPV', 'trk_ptErr',
                     'trk_etaErr', 'trk_dxyErr', 'trk_dzErr', 'trk_nChi2', 'trk_ndof', 'trk_nLost', 'trk_nPixel', 'trk_nStrip',
                     'trk_algo']
    if "QCD" in listOfNames[0]:
        dataframe = read_root(listOfNames, columns=columnsToRead, flatten=columnsToRead)
    else:
        dataframe = read_root(listOfNames, columns=columnsToRead)

    return dataframe
