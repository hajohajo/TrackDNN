import os
import shutil

#v
inputVariables = ['trk_pt', 'trk_eta', 'trk_dxy', 'trk_dz', 'trk_dxyClosestPV', 'trk_dzClosestPV', 'trk_ptErr',
                  'trk_etaErr', 'trk_dxyErr', 'trk_dzErr', 'trk_nChi2', 'trk_ndof', 'trk_nLost', 'trk_nPixel',
                  'trk_nStrip']


def createFolders(listOfFolders):
    for dir in listOfFolders:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

#Convenience class for switching between trk_algo indices and step names
class _Enum:
    def __init__(self, **values):
        self._reverse = {}
        for key, value in values.items():
            setattr(self, key, value)
            if value in self._reverse:
                raise Exception("Value %s is already used for a key %s, tried to re-add it for key %s" % (
                value, self._reverse[value], key))
            self._reverse[value] = key

    def toString(self, val):
        return self._reverse[val]


Algo = _Enum(
    undefAlgorithm=0, ctf=1,
    duplicateMerge=2, cosmics=3,
    initialStep=4,
    lowPtTripletStep=5,
    pixelPairStep=6,
    detachedTripletStep=7,
    mixedTripletStep=8,
    pixelLessStep=9,
    tobTecStep=10,
    jetCoreRegionalStep=11,
    conversionStep=12,
    muonSeededStepInOut=13,
    muonSeededStepOutIn=14,
    outInEcalSeededConv=15, inOutEcalSeededConv=16,
    nuclInter=17,
    standAloneMuon=18, globalMuon=19, cosmicStandAloneMuon=20,
    cosmicGlobalMuon=21,
    # Phase1
    highPtTripletStep=22, lowPtQuadStep=23, detachedQuadStep=24,
    reservedForUpgrades1=25, reservedForUpgrades2=26,
    bTagGhostTracks=27,
    beamhalo=28,
    gsf=29,
    # HLT algo name
    hltPixel=30,
    # steps used by PF
    hltIter0=31,
    hltIter1=32,
    hltIter2=33,
    hltIter3=34,
    hltIter4=35,
    # steps used by all other objects @HLT
    hltIterX=36,
    # steps used by HI muon regional iterative tracking
    hiRegitMuInitialStep=37,
    hiRegitMuLowPtTripletStep=38,
    hiRegitMuPixelPairStep=39,
    hiRegitMuDetachedTripletStep=40,
    hiRegitMuMixedTripletStep=41,
    hiRegitMuPixelLessStep=42,
    hiRegitMuTobTecStep=43,
    hiRegitMuMuonSeededStepInOut=44,
    hiRegitMuMuonSeededStepOutIn=45,
    algoSize=46,
    allSteps=99
)