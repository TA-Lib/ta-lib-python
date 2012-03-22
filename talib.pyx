"""
Cython interface to the ta-lib TA_MA function
Author : Didrik Pinte <dpinte@enthought.com>
Reference : http://ta-lib.org
"""

import numpy
cimport numpy as np

ctypedef int TA_RetCode 
# extract the needed part of ta_libc.h that I will use in the inerface
cdef extern from "ta_libc.h":
    enum: TA_SUCCESS
    # ! can't use const in function declaration (cython 0.12 restriction) - just removing them does the trick
    TA_RetCode TA_MA(int startIdx, int endIdx, double inReal[], int optInTimePeriod, int optInMAType, int *outBegIdx, int *outNbElement, double outReal[]) 
    TA_RetCode TA_KAMA(int startIdx, int endIdx, double inReal[], int optInTimePeriod, int *outBegIdx, int *outNbElement, double outReal[]) 
    TA_RetCode TA_BBANDS(int startIdx, int endIdx, double inReal[], int optInTimePeriod, double optInNbDevUp, double optInNbDevDn, int optInMAType, int *outBegIdx, int *outNbElement, double outRealUpperBand[], double outRealMiddleBand[], double outRealLowerBand[])
    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()

def moving_average(np.ndarray[np.float_t, ndim=1] inreal,
               int begIdx=0, int endIdx=-1,
               int optInTimePeriod=10, int optInMAType=1):

    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros_like(inreal)

    if endIdx == -1:
        endIdx = inreal.shape[0]-1

    retCode =  TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!\n" % retCode)
    else:
        retCode =  TA_MA(begIdx, endIdx, <double *>inreal.data,
                         optInTimePeriod, optInMAType, &outBegIdx,
                         &outNBElement, <double *>outreal.data)
    TA_Shutdown()

    return (outBegIdx, outNBElement, outreal)


def bollinger_bands(np.ndarray[np.float_t, ndim=1] inreal,
               int begIdx=0, int endIdx=-1,
               int optInTimePeriod=10,
               double optInNbDevUp=1, double optInNbDevDn=1,
               int optInMAType=0):

    cdef int outBegIdx
    cdef int outNBElement

    cdef np.ndarray[np.float_t, ndim=1] outRealUpperBand = numpy.zeros_like(inreal)
    cdef np.ndarray[np.float_t, ndim=1] outRealMiddleBand = numpy.zeros_like(inreal)
    cdef np.ndarray[np.float_t, ndim=1] outRealLowerBand = numpy.zeros_like(inreal)

    if endIdx == -1:
        endIdx = inreal.shape[0]-1

    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!\n" % retCode)
    else:
        retCode = TA_BBANDS(begIdx, endIdx, <double *>inreal.data,
                            optInTimePeriod, optInNbDevUp, optInNbDevDn,
                            optInMAType, &outBegIdx, &outNBElement,
                            <double *>outRealUpperBand.data,
                            <double *>outRealMiddleBand.data,
                            <double *>outRealLowerBand.data)
    TA_Shutdown()

    return (outBegIdx, outNBElement, outRealUpperBand, outRealMiddleBand, outRealLowerBand)


def kama(np.ndarray[np.float_t, ndim=1] inreal,
        int begIdx=0, int endIdx=-1,
        int optInTimePeriod=10):

    cdef int outBegIdx
    cdef int outNBElement

    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros_like(inreal)

    if endIdx == -1:
        endIdx = inreal.shape[0]-1

    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!\n" % retCode)
    else:
        retCode =  TA_KAMA(begIdx, endIdx, <double *>inreal.data,
                           optInTimePeriod, &outBegIdx, &outNBElement,
                           <double *>outreal.data)
    TA_Shutdown()

    return (outBegIdx, outNBElement, outreal)

