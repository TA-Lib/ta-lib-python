cimport numpy as np
from numpy import nan
from cython import boundscheck, wraparound

from .common cimport _ta_check_success

cdef double NaN = nan

cdef extern from "numpy/arrayobject.h":
    int PyArray_TYPE(np.ndarray)
    object PyArray_EMPTY(int, np.npy_intp*, int, int)
    int PyArray_FLAGS(np.ndarray)
    object PyArray_GETCONTIGUOUS(np.ndarray)

np.import_array() # Initialize the NumPy C API

cimport libta_lib as lib
from libta_lib cimport TA_RetCode

lib.TA_Initialize()

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ACOS( np.ndarray real not None ):
    """ ACOS(real)

    Vector Trigonometric ACos (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ACOS( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ACOS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AD( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None ):
    """ AD(high, low, close, volume)

    Chaikin A/D Line (Volume Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    if PyArray_TYPE(volume) != np.NPY_DOUBLE:
        raise Exception("volume is not double")
    if volume.ndim != 1:
        raise Exception("volume has wrong dimensions")
    if not (PyArray_FLAGS(volume) & np.NPY_C_CONTIGUOUS):
        volume = PyArray_GETCONTIGUOUS(volume)
    volume_data = <double*>volume.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    if length != volume.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_AD( length - 1 , length - 1 , high_data , low_data , close_data , volume_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADD( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ ADD(real0, real1)

    Vector Arithmetic Add (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real0) != np.NPY_DOUBLE:
        raise Exception("real0 is not double")
    if real0.ndim != 1:
        raise Exception("real0 has wrong dimensions")
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    if PyArray_TYPE(real1) != np.NPY_DOUBLE:
        raise Exception("real1 is not double")
    if real1.ndim != 1:
        raise Exception("real1 has wrong dimensions")
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    if length != real1.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_ADD( length - 1 , length - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int fastperiod=-2**31 , int slowperiod=-2**31 ):
    """ ADOSC(high, low, close, volume[, fastperiod=?, slowperiod=?])

    Chaikin A/D Oscillator (Volume Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Parameters:
        fastperiod: 3
        slowperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    if PyArray_TYPE(volume) != np.NPY_DOUBLE:
        raise Exception("volume is not double")
    if volume.ndim != 1:
        raise Exception("volume has wrong dimensions")
    if not (PyArray_FLAGS(volume) & np.NPY_C_CONTIGUOUS):
        volume = PyArray_GETCONTIGUOUS(volume)
    volume_data = <double*>volume.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    if length != volume.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_ADOSC( length - 1 , length - 1 , high_data , low_data , close_data , volume_data , fastperiod , slowperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ADX(high, low, close[, timeperiod=?])

    Average Directional Movement Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_ADX( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADXR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ADXR(high, low, close[, timeperiod=?])

    Average Directional Movement Index Rating (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_ADXR( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADXR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def APO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
    """ APO(real[, fastperiod=?, slowperiod=?, matype=?])

    Absolute Price Oscillator (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_APO( length - 1 , length - 1 , real_data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_APO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AROON( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ AROON(high, low[, timeperiod=?])

    Aroon (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        aroondown
        aroonup
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outaroondown
        double outaroonup
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    outaroondown = NaN
    outaroonup = NaN
    retCode = lib.TA_AROON( length - 1 , length - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outaroondown , &outaroonup )
    _ta_check_success("TA_AROON", retCode)
    return outaroondown , outaroonup 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AROONOSC( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ AROONOSC(high, low[, timeperiod=?])

    Aroon Oscillator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_AROONOSC( length - 1 , length - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AROONOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ASIN( np.ndarray real not None ):
    """ ASIN(real)

    Vector Trigonometric ASin (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ASIN( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ASIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ATAN( np.ndarray real not None ):
    """ ATAN(real)

    Vector Trigonometric ATan (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ATAN( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ATAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ ATR(high, low, close[, timeperiod=?])

    Average True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_ATR( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AVGPRICE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ AVGPRICE(open, high, low, close)

    Average Price (Price Transform)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_AVGPRICE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AVGPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BBANDS( np.ndarray real not None , int timeperiod=-2**31 , double nbdevup=-4e37 , double nbdevdn=-4e37 , int matype=0 ):
    """ BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])

    Bollinger Bands (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdevup: 2
        nbdevdn: 2
        matype: 0 (Simple Moving Average)
    Outputs:
        upperband
        middleband
        lowerband
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outrealupperband
        double outrealmiddleband
        double outreallowerband
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outrealupperband = NaN
    outrealmiddleband = NaN
    outreallowerband = NaN
    retCode = lib.TA_BBANDS( length - 1 , length - 1 , real_data , timeperiod , nbdevup , nbdevdn , matype , &outbegidx , &outnbelement , &outrealupperband , &outrealmiddleband , &outreallowerband )
    _ta_check_success("TA_BBANDS", retCode)
    return outrealupperband , outrealmiddleband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BETA( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
    """ BETA(real0, real1[, timeperiod=?])

    Beta (Statistic Functions)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Parameters:
        timeperiod: 5
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real0) != np.NPY_DOUBLE:
        raise Exception("real0 is not double")
    if real0.ndim != 1:
        raise Exception("real0 has wrong dimensions")
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    if PyArray_TYPE(real1) != np.NPY_DOUBLE:
        raise Exception("real1 is not double")
    if real1.ndim != 1:
        raise Exception("real1 has wrong dimensions")
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    if length != real1.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_BETA( length - 1 , length - 1 , real0_data , real1_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_BETA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BOP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ BOP(open, high, low, close)

    Balance Of Power (Momentum Indicators)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_BOP( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_BOP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CCI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ CCI(high, low, close[, timeperiod=?])

    Commodity Channel Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_CCI( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CCI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL2CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL2CROWS(open, high, low, close)

    Two Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDL2CROWS( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL2CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3BLACKCROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3BLACKCROWS(open, high, low, close)

    Three Black Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDL3BLACKCROWS( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3BLACKCROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3INSIDE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3INSIDE(open, high, low, close)

    Three Inside Up/Down (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDL3INSIDE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3INSIDE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3LINESTRIKE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3LINESTRIKE(open, high, low, close)

    Three-Line Strike  (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDL3LINESTRIKE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3LINESTRIKE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3OUTSIDE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3OUTSIDE(open, high, low, close)

    Three Outside Up/Down (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDL3OUTSIDE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3OUTSIDE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3STARSINSOUTH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3STARSINSOUTH(open, high, low, close)

    Three Stars In The South (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDL3STARSINSOUTH( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3STARSINSOUTH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3WHITESOLDIERS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3WHITESOLDIERS(open, high, low, close)

    Three Advancing White Soldiers (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDL3WHITESOLDIERS( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3WHITESOLDIERS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLABANDONEDBABY( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
    """ CDLABANDONEDBABY(open, high, low, close[, penetration=?])

    Abandoned Baby (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLABANDONEDBABY( length - 1 , length - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLABANDONEDBABY", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLADVANCEBLOCK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLADVANCEBLOCK(open, high, low, close)

    Advance Block (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLADVANCEBLOCK( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLADVANCEBLOCK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLBELTHOLD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLBELTHOLD(open, high, low, close)

    Belt-hold (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLBELTHOLD( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLBELTHOLD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLBREAKAWAY( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLBREAKAWAY(open, high, low, close)

    Breakaway (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLBREAKAWAY( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLBREAKAWAY", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLCLOSINGMARUBOZU( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLCLOSINGMARUBOZU(open, high, low, close)

    Closing Marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLCLOSINGMARUBOZU( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLCLOSINGMARUBOZU", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLCONCEALBABYSWALL( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLCONCEALBABYSWALL(open, high, low, close)

    Concealing Baby Swallow (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLCONCEALBABYSWALL( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLCONCEALBABYSWALL", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLCOUNTERATTACK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLCOUNTERATTACK(open, high, low, close)

    Counterattack (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLCOUNTERATTACK( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLCOUNTERATTACK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLDARKCLOUDCOVER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.5 ):
    """ CDLDARKCLOUDCOVER(open, high, low, close[, penetration=?])

    Dark Cloud Cover (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.5
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLDARKCLOUDCOVER( length - 1 , length - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLDARKCLOUDCOVER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLDOJI(open, high, low, close)

    Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLDOJI( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLDOJISTAR(open, high, low, close)

    Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLDOJISTAR( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLDRAGONFLYDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLDRAGONFLYDOJI(open, high, low, close)

    Dragonfly Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLDRAGONFLYDOJI( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLDRAGONFLYDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLENGULFING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLENGULFING(open, high, low, close)

    Engulfing Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLENGULFING( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLENGULFING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLEVENINGDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
    """ CDLEVENINGDOJISTAR(open, high, low, close[, penetration=?])

    Evening Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLEVENINGDOJISTAR( length - 1 , length - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLEVENINGDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLEVENINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
    """ CDLEVENINGSTAR(open, high, low, close[, penetration=?])

    Evening Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLEVENINGSTAR( length - 1 , length - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLEVENINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLGAPSIDESIDEWHITE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLGAPSIDESIDEWHITE(open, high, low, close)

    Up/Down-gap side-by-side white lines (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLGAPSIDESIDEWHITE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLGAPSIDESIDEWHITE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLGRAVESTONEDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLGRAVESTONEDOJI(open, high, low, close)

    Gravestone Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLGRAVESTONEDOJI( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLGRAVESTONEDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHAMMER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHAMMER(open, high, low, close)

    Hammer (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLHAMMER( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHAMMER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHANGINGMAN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHANGINGMAN(open, high, low, close)

    Hanging Man (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLHANGINGMAN( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHANGINGMAN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHARAMI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHARAMI(open, high, low, close)

    Harami Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLHARAMI( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHARAMI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHARAMICROSS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHARAMICROSS(open, high, low, close)

    Harami Cross Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLHARAMICROSS( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHARAMICROSS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHIGHWAVE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHIGHWAVE(open, high, low, close)

    High-Wave Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLHIGHWAVE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHIGHWAVE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHIKKAKE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHIKKAKE(open, high, low, close)

    Hikkake Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLHIKKAKE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHIKKAKE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHIKKAKEMOD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHIKKAKEMOD(open, high, low, close)

    Modified Hikkake Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLHIKKAKEMOD( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHIKKAKEMOD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHOMINGPIGEON( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHOMINGPIGEON(open, high, low, close)

    Homing Pigeon (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLHOMINGPIGEON( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHOMINGPIGEON", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLIDENTICAL3CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLIDENTICAL3CROWS(open, high, low, close)

    Identical Three Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLIDENTICAL3CROWS( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLIDENTICAL3CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLINNECK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLINNECK(open, high, low, close)

    In-Neck Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLINNECK( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLINNECK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLINVERTEDHAMMER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLINVERTEDHAMMER(open, high, low, close)

    Inverted Hammer (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLINVERTEDHAMMER( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLINVERTEDHAMMER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLKICKING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLKICKING(open, high, low, close)

    Kicking (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLKICKING( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLKICKING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLKICKINGBYLENGTH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLKICKINGBYLENGTH(open, high, low, close)

    Kicking - bull/bear determined by the longer marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLKICKINGBYLENGTH( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLKICKINGBYLENGTH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLLADDERBOTTOM( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLLADDERBOTTOM(open, high, low, close)

    Ladder Bottom (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLLADDERBOTTOM( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLLADDERBOTTOM", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLLONGLEGGEDDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLLONGLEGGEDDOJI(open, high, low, close)

    Long Legged Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLLONGLEGGEDDOJI( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLLONGLEGGEDDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLLONGLINE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLLONGLINE(open, high, low, close)

    Long Line Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLLONGLINE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLLONGLINE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMARUBOZU( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLMARUBOZU(open, high, low, close)

    Marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLMARUBOZU( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMARUBOZU", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMATCHINGLOW( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLMATCHINGLOW(open, high, low, close)

    Matching Low (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLMATCHINGLOW( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMATCHINGLOW", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMATHOLD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.5 ):
    """ CDLMATHOLD(open, high, low, close[, penetration=?])

    Mat Hold (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.5
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLMATHOLD( length - 1 , length - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMATHOLD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMORNINGDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
    """ CDLMORNINGDOJISTAR(open, high, low, close[, penetration=?])

    Morning Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLMORNINGDOJISTAR( length - 1 , length - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMORNINGDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMORNINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
    """ CDLMORNINGSTAR(open, high, low, close[, penetration=?])

    Morning Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLMORNINGSTAR( length - 1 , length - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMORNINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLONNECK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLONNECK(open, high, low, close)

    On-Neck Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLONNECK( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLONNECK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLPIERCING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLPIERCING(open, high, low, close)

    Piercing Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLPIERCING( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLPIERCING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLRICKSHAWMAN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLRICKSHAWMAN(open, high, low, close)

    Rickshaw Man (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLRICKSHAWMAN( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLRICKSHAWMAN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLRISEFALL3METHODS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLRISEFALL3METHODS(open, high, low, close)

    Rising/Falling Three Methods (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLRISEFALL3METHODS( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLRISEFALL3METHODS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSEPARATINGLINES( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSEPARATINGLINES(open, high, low, close)

    Separating Lines (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLSEPARATINGLINES( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSEPARATINGLINES", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSHOOTINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSHOOTINGSTAR(open, high, low, close)

    Shooting Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLSHOOTINGSTAR( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSHOOTINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSHORTLINE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSHORTLINE(open, high, low, close)

    Short Line Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLSHORTLINE( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSHORTLINE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSPINNINGTOP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSPINNINGTOP(open, high, low, close)

    Spinning Top (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLSPINNINGTOP( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSPINNINGTOP", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSTALLEDPATTERN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSTALLEDPATTERN(open, high, low, close)

    Stalled Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLSTALLEDPATTERN( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSTALLEDPATTERN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSTICKSANDWICH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSTICKSANDWICH(open, high, low, close)

    Stick Sandwich (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLSTICKSANDWICH( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSTICKSANDWICH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLTAKURI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLTAKURI(open, high, low, close)

    Takuri (Dragonfly Doji with very long lower shadow) (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLTAKURI( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLTAKURI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLTASUKIGAP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLTASUKIGAP(open, high, low, close)

    Tasuki Gap (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLTASUKIGAP( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLTASUKIGAP", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLTHRUSTING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLTHRUSTING(open, high, low, close)

    Thrusting Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLTHRUSTING( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLTHRUSTING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLTRISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLTRISTAR(open, high, low, close)

    Tristar Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLTRISTAR( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLTRISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLUNIQUE3RIVER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLUNIQUE3RIVER(open, high, low, close)

    Unique 3 River (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLUNIQUE3RIVER( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLUNIQUE3RIVER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLUPSIDEGAP2CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLUPSIDEGAP2CROWS(open, high, low, close)

    Upside Gap Two Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLUPSIDEGAP2CROWS( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLUPSIDEGAP2CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLXSIDEGAP3METHODS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLXSIDEGAP3METHODS(open, high, low, close)

    Upside/Downside Gap Three Methods (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(open) != np.NPY_DOUBLE:
        raise Exception("open is not double")
    if open.ndim != 1:
        raise Exception("open has wrong dimensions")
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = open.shape[0]
    if length != high.shape[0]:
        raise Exception("input lengths are different")
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outinteger = 0
    retCode = lib.TA_CDLXSIDEGAP3METHODS( length - 1 , length - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLXSIDEGAP3METHODS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CEIL( np.ndarray real not None ):
    """ CEIL(real)

    Vector Ceil (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_CEIL( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CEIL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CMO( np.ndarray real not None , int timeperiod=-2**31 ):
    """ CMO(real[, timeperiod=?])

    Chande Momentum Oscillator (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_CMO( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CMO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CORREL( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
    """ CORREL(real0, real1[, timeperiod=?])

    Pearson's Correlation Coefficient (r) (Statistic Functions)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real0) != np.NPY_DOUBLE:
        raise Exception("real0 is not double")
    if real0.ndim != 1:
        raise Exception("real0 has wrong dimensions")
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    if PyArray_TYPE(real1) != np.NPY_DOUBLE:
        raise Exception("real1 is not double")
    if real1.ndim != 1:
        raise Exception("real1 has wrong dimensions")
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    if length != real1.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_CORREL( length - 1 , length - 1 , real0_data , real1_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CORREL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def COS( np.ndarray real not None ):
    """ COS(real)

    Vector Trigonometric Cos (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_COS( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_COS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def COSH( np.ndarray real not None ):
    """ COSH(real)

    Vector Trigonometric Cosh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_COSH( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_COSH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DEMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ DEMA(real[, timeperiod=?])

    Double Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_DEMA( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DIV( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ DIV(real0, real1)

    Vector Arithmetic Div (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real0) != np.NPY_DOUBLE:
        raise Exception("real0 is not double")
    if real0.ndim != 1:
        raise Exception("real0 has wrong dimensions")
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    if PyArray_TYPE(real1) != np.NPY_DOUBLE:
        raise Exception("real1 is not double")
    if real1.ndim != 1:
        raise Exception("real1 has wrong dimensions")
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    if length != real1.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_DIV( length - 1 , length - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DIV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ DX(high, low, close[, timeperiod=?])

    Directional Movement Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_DX( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def EMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ EMA(real[, timeperiod=?])

    Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_EMA( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_EMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def EXP( np.ndarray real not None ):
    """ EXP(real)

    Vector Arithmetic Exp (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_EXP( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_EXP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def FLOOR( np.ndarray real not None ):
    """ FLOOR(real)

    Vector Floor (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_FLOOR( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_FLOOR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_DCPERIOD( np.ndarray real not None ):
    """ HT_DCPERIOD(real)

    Hilbert Transform - Dominant Cycle Period (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_DCPERIOD( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_DCPERIOD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_DCPHASE( np.ndarray real not None ):
    """ HT_DCPHASE(real)

    Hilbert Transform - Dominant Cycle Phase (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_DCPHASE( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_DCPHASE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_PHASOR( np.ndarray real not None ):
    """ HT_PHASOR(real)

    Hilbert Transform - Phasor Components (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        inphase
        quadrature
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outinphase
        double outquadrature
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinphase = NaN
    outquadrature = NaN
    retCode = lib.TA_HT_PHASOR( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outinphase , &outquadrature )
    _ta_check_success("TA_HT_PHASOR", retCode)
    return outinphase , outquadrature 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_SINE( np.ndarray real not None ):
    """ HT_SINE(real)

    Hilbert Transform - SineWave (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        sine
        leadsine
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outsine
        double outleadsine
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outsine = NaN
    outleadsine = NaN
    retCode = lib.TA_HT_SINE( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outsine , &outleadsine )
    _ta_check_success("TA_HT_SINE", retCode)
    return outsine , outleadsine 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_TRENDLINE( np.ndarray real not None ):
    """ HT_TRENDLINE(real)

    Hilbert Transform - Instantaneous Trendline (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_TRENDLINE( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_TRENDLINE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_TRENDMODE( np.ndarray real not None ):
    """ HT_TRENDMODE(real)

    Hilbert Transform - Trend vs Cycle Mode (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_HT_TRENDMODE( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_HT_TRENDMODE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def KAMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ KAMA(real[, timeperiod=?])

    Kaufman Adaptive Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_KAMA( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_KAMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG(real[, timeperiod=?])

    Linear Regression (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_ANGLE( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_ANGLE(real[, timeperiod=?])

    Linear Regression Angle (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_ANGLE( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_ANGLE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_INTERCEPT( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_INTERCEPT(real[, timeperiod=?])

    Linear Regression Intercept (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_INTERCEPT( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_INTERCEPT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_SLOPE( np.ndarray real not None , int timeperiod=-2**31 ):
    """ LINEARREG_SLOPE(real[, timeperiod=?])

    Linear Regression Slope (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_SLOPE( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_SLOPE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LN( np.ndarray real not None ):
    """ LN(real)

    Vector Log Natural (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LN( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LOG10( np.ndarray real not None ):
    """ LOG10(real)

    Vector Log10 (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LOG10( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LOG10", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MA( np.ndarray real not None , int timeperiod=-2**31 , int matype=0 ):
    """ MA(real[, timeperiod=?, matype=?])

    Moving average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MA( length - 1 , length - 1 , real_data , timeperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACD( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int signalperiod=-2**31 ):
    """ MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])

    Moving Average Convergence/Divergence (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        signalperiod: 9
    Outputs:
        macd
        macdsignal
        macdhist
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACD( length - 1 , length - 1 , real_data , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACD", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACDEXT( np.ndarray real not None , int fastperiod=-2**31 , int fastmatype=0 , int slowperiod=-2**31 , int slowmatype=0 , int signalperiod=-2**31 , int signalmatype=0 ):
    """ MACDEXT(real[, fastperiod=?, fastmatype=?, slowperiod=?, slowmatype=?, signalperiod=?, signalmatype=?])

    MACD with controllable MA type (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        fastmatype: 0
        slowperiod: 26
        slowmatype: 0
        signalperiod: 9
        signalmatype: 0
    Outputs:
        macd
        macdsignal
        macdhist
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACDEXT( length - 1 , length - 1 , real_data , fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACDEXT", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACDFIX( np.ndarray real not None , int signalperiod=-2**31 ):
    """ MACDFIX(real[, signalperiod=?])

    Moving Average Convergence/Divergence Fix 12/26 (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        signalperiod: 9
    Outputs:
        macd
        macdsignal
        macdhist
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACDFIX( length - 1 , length - 1 , real_data , signalperiod , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACDFIX", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAMA( np.ndarray real not None , double fastlimit=-4e37 , double slowlimit=-4e37 ):
    """ MAMA(real[, fastlimit=?, slowlimit=?])

    MESA Adaptive Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastlimit: 0.5
        slowlimit: 0.05
    Outputs:
        mama
        fama
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmama
        double outfama
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmama = NaN
    outfama = NaN
    retCode = lib.TA_MAMA( length - 1 , length - 1 , real_data , fastlimit , slowlimit , &outbegidx , &outnbelement , &outmama , &outfama )
    _ta_check_success("TA_MAMA", retCode)
    return outmama , outfama 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAVP( np.ndarray real not None , np.ndarray periods not None , int minperiod=-2**31 , int maxperiod=-2**31 , int matype=0 ):
    """ MAVP(real, periods[, minperiod=?, maxperiod=?, matype=?])

    Moving average with variable period (Overlap Studies)

    Inputs:
        real: (any ndarray)
        periods: (any ndarray)
    Parameters:
        minperiod: 2
        maxperiod: 30
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        double* periods_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    if PyArray_TYPE(periods) != np.NPY_DOUBLE:
        raise Exception("periods is not double")
    if periods.ndim != 1:
        raise Exception("periods has wrong dimensions")
    if not (PyArray_FLAGS(periods) & np.NPY_C_CONTIGUOUS):
        periods = PyArray_GETCONTIGUOUS(periods)
    periods_data = <double*>periods.data
    length = real.shape[0]
    if length != periods.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_MAVP( length - 1 , length - 1 , real_data , periods_data , minperiod , maxperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MAVP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MAX(real[, timeperiod=?])

    Highest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MAX( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MAX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MAXINDEX(real[, timeperiod=?])

    Index of highest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_MAXINDEX( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_MAXINDEX", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MEDPRICE( np.ndarray high not None , np.ndarray low not None ):
    """ MEDPRICE(high, low)

    Median Price (Price Transform)

    Inputs:
        prices: ['high', 'low']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_MEDPRICE( length - 1 , length - 1 , high_data , low_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MEDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MFI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int timeperiod=-2**31 ):
    """ MFI(high, low, close, volume[, timeperiod=?])

    Money Flow Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    if PyArray_TYPE(volume) != np.NPY_DOUBLE:
        raise Exception("volume is not double")
    if volume.ndim != 1:
        raise Exception("volume has wrong dimensions")
    if not (PyArray_FLAGS(volume) & np.NPY_C_CONTIGUOUS):
        volume = PyArray_GETCONTIGUOUS(volume)
    volume_data = <double*>volume.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    if length != volume.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_MFI( length - 1 , length - 1 , high_data , low_data , close_data , volume_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MFI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIDPOINT( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MIDPOINT(real[, timeperiod=?])

    MidPoint over period (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MIDPOINT( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIDPOINT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIDPRICE( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ MIDPRICE(high, low[, timeperiod=?])

    Midpoint Price over period (Overlap Studies)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_MIDPRICE( length - 1 , length - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIN( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MIN(real[, timeperiod=?])

    Lowest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MIN( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MININDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MININDEX(real[, timeperiod=?])

    Index of lowest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_MININDEX( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_MININDEX", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINMAX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MINMAX(real[, timeperiod=?])

    Lowest and highest values over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        min
        max
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmin
        double outmax
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmin = NaN
    outmax = NaN
    retCode = lib.TA_MINMAX( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outmin , &outmax )
    _ta_check_success("TA_MINMAX", retCode)
    return outmin , outmax 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINMAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MINMAXINDEX(real[, timeperiod=?])

    Indexes of lowest and highest values over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        minidx
        maxidx
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outminidx
        int outmaxidx
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outminidx = 0
    outmaxidx = 0
    retCode = lib.TA_MINMAXINDEX( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outminidx , &outmaxidx )
    _ta_check_success("TA_MINMAXINDEX", retCode)
    return outminidx , outmaxidx 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ MINUS_DI(high, low, close[, timeperiod=?])

    Minus Directional Indicator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_MINUS_DI( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MINUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ MINUS_DM(high, low[, timeperiod=?])

    Minus Directional Movement (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_MINUS_DM( length - 1 , length - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MINUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MOM( np.ndarray real not None , int timeperiod=-2**31 ):
    """ MOM(real[, timeperiod=?])

    Momentum (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MOM( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MOM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MULT( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ MULT(real0, real1)

    Vector Arithmetic Mult (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real0) != np.NPY_DOUBLE:
        raise Exception("real0 is not double")
    if real0.ndim != 1:
        raise Exception("real0 has wrong dimensions")
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    if PyArray_TYPE(real1) != np.NPY_DOUBLE:
        raise Exception("real1 is not double")
    if real1.ndim != 1:
        raise Exception("real1 has wrong dimensions")
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    if length != real1.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_MULT( length - 1 , length - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MULT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def NATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ NATR(high, low, close[, timeperiod=?])

    Normalized Average True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_NATR( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_NATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def OBV( np.ndarray real not None , np.ndarray volume not None ):
    """ OBV(real, volume)

    On Balance Volume (Volume Indicators)

    Inputs:
        real: (any ndarray)
        prices: ['volume']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    if PyArray_TYPE(volume) != np.NPY_DOUBLE:
        raise Exception("volume is not double")
    if volume.ndim != 1:
        raise Exception("volume has wrong dimensions")
    if not (PyArray_FLAGS(volume) & np.NPY_C_CONTIGUOUS):
        volume = PyArray_GETCONTIGUOUS(volume)
    volume_data = <double*>volume.data
    length = real.shape[0]
    if length != volume.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_OBV( length - 1 , length - 1 , real_data , volume_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_OBV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PLUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ PLUS_DI(high, low, close[, timeperiod=?])

    Plus Directional Indicator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_PLUS_DI( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PLUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PLUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """ PLUS_DM(high, low[, timeperiod=?])

    Plus Directional Movement (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_PLUS_DM( length - 1 , length - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PLUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PPO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
    """ PPO(real[, fastperiod=?, slowperiod=?, matype=?])

    Percentage Price Oscillator (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_PPO( length - 1 , length - 1 , real_data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PPO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROC( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROC(real[, timeperiod=?])

    Rate of change : ((real/prevPrice)-1)*100 (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROC( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCP( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCP(real[, timeperiod=?])

    Rate of change Percentage: (real-prevPrice)/prevPrice (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCP( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCR( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCR(real[, timeperiod=?])

    Rate of change ratio: (real/prevPrice) (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCR( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCR100( np.ndarray real not None , int timeperiod=-2**31 ):
    """ ROCR100(real[, timeperiod=?])

    Rate of change ratio 100 scale: (real/prevPrice)*100 (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCR100( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCR100", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def RSI( np.ndarray real not None , int timeperiod=-2**31 ):
    """ RSI(real[, timeperiod=?])

    Relative Strength Index (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_RSI( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_RSI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SAR( np.ndarray high not None , np.ndarray low not None , double acceleration=0.02 , double maximum=0.2 ):
    """ SAR(high, low[, acceleration=?, maximum=?])

    Parabolic SAR (Overlap Studies)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        acceleration: 0.02
        maximum: 0.2
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_SAR( length - 1 , length - 1 , high_data , low_data , acceleration , maximum , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SAREXT( np.ndarray high not None , np.ndarray low not None , double startvalue=-4e37 , double offsetonreverse=-4e37 , double accelerationinitlong=-4e37 , double accelerationlong=-4e37 , double accelerationmaxlong=-4e37 , double accelerationinitshort=-4e37 , double accelerationshort=-4e37 , double accelerationmaxshort=-4e37 ):
    """ SAREXT(high, low[, startvalue=?, offsetonreverse=?, accelerationinitlong=?, accelerationlong=?, accelerationmaxlong=?, accelerationinitshort=?, accelerationshort=?, accelerationmaxshort=?])

    Parabolic SAR - Extended (Overlap Studies)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        startvalue: 0
        offsetonreverse: 0
        accelerationinitlong: 0.02
        accelerationlong: 0.02
        accelerationmaxlong: 0.2
        accelerationinitshort: 0.02
        accelerationshort: 0.02
        accelerationmaxshort: 0.2
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_SAREXT( length - 1 , length - 1 , high_data , low_data , startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SAREXT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SIN( np.ndarray real not None ):
    """ SIN(real)

    Vector Trigonometric Sin (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SIN( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SINH( np.ndarray real not None ):
    """ SINH(real)

    Vector Trigonometric Sinh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SINH( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SINH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ SMA(real[, timeperiod=?])

    Simple Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SMA( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SQRT( np.ndarray real not None ):
    """ SQRT(real)

    Vector Square Root (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SQRT( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SQRT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STDDEV( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """ STDDEV(real[, timeperiod=?, nbdev=?])

    Standard Deviation (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdev: 1
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_STDDEV( length - 1 , length - 1 , real_data , timeperiod , nbdev , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_STDDEV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCH( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int slowk_period=-2**31 , int slowk_matype=0 , int slowd_period=-2**31 , int slowd_matype=0 ):
    """ STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])

    Stochastic (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 5
        slowk_period: 3
        slowk_matype: 0
        slowd_period: 3
        slowd_matype: 0
    Outputs:
        slowk
        slowd
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outslowk
        double outslowd
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outslowk = NaN
    outslowd = NaN
    retCode = lib.TA_STOCH( length - 1 , length - 1 , high_data , low_data , close_data , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , &outslowk , &outslowd )
    _ta_check_success("TA_STOCH", retCode)
    return outslowk , outslowd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCHF( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
    """ STOCHF(high, low, close[, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Fast (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 5
        fastd_period: 3
        fastd_matype: 0
    Outputs:
        fastk
        fastd
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outfastk
        double outfastd
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outfastk = NaN
    outfastd = NaN
    retCode = lib.TA_STOCHF( length - 1 , length - 1 , high_data , low_data , close_data , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , &outfastk , &outfastd )
    _ta_check_success("TA_STOCHF", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCHRSI( np.ndarray real not None , int timeperiod=-2**31 , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
    """ STOCHRSI(real[, timeperiod=?, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Relative Strength Index (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
        fastk_period: 5
        fastd_period: 3
        fastd_matype: 0
    Outputs:
        fastk
        fastd
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outfastk
        double outfastd
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outfastk = NaN
    outfastd = NaN
    retCode = lib.TA_STOCHRSI( length - 1 , length - 1 , real_data , timeperiod , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , &outfastk , &outfastd )
    _ta_check_success("TA_STOCHRSI", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SUB( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ SUB(real0, real1)

    Vector Arithmetic Substraction (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real0) != np.NPY_DOUBLE:
        raise Exception("real0 is not double")
    if real0.ndim != 1:
        raise Exception("real0 has wrong dimensions")
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    if PyArray_TYPE(real1) != np.NPY_DOUBLE:
        raise Exception("real1 is not double")
    if real1.ndim != 1:
        raise Exception("real1 has wrong dimensions")
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    if length != real1.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_SUB( length - 1 , length - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SUB", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SUM( np.ndarray real not None , int timeperiod=-2**31 ):
    """ SUM(real[, timeperiod=?])

    Summation (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SUM( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SUM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def T3( np.ndarray real not None , int timeperiod=-2**31 , double vfactor=-4e37 ):
    """ T3(real[, timeperiod=?, vfactor=?])

    Triple Exponential Moving Average (T3) (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        vfactor: 0.7
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_T3( length - 1 , length - 1 , real_data , timeperiod , vfactor , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_T3", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TAN( np.ndarray real not None ):
    """ TAN(real)

    Vector Trigonometric Tan (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TAN( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TANH( np.ndarray real not None ):
    """ TANH(real)

    Vector Trigonometric Tanh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TANH( length - 1 , length - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TANH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TEMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TEMA(real[, timeperiod=?])

    Triple Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TEMA( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TRANGE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ TRANGE(high, low, close)

    True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_TRANGE( length - 1 , length - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRANGE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TRIMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TRIMA(real[, timeperiod=?])

    Triangular Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TRIMA( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRIMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TRIX( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TRIX(real[, timeperiod=?])

    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TRIX( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRIX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TSF( np.ndarray real not None , int timeperiod=-2**31 ):
    """ TSF(real[, timeperiod=?])

    Time Series Forecast (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TSF( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TSF", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TYPPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ TYPPRICE(high, low, close)

    Typical Price (Price Transform)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_TYPPRICE( length - 1 , length - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TYPPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ULTOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod1=-2**31 , int timeperiod2=-2**31 , int timeperiod3=-2**31 ):
    """ ULTOSC(high, low, close[, timeperiod1=?, timeperiod2=?, timeperiod3=?])

    Ultimate Oscillator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod1: 7
        timeperiod2: 14
        timeperiod3: 28
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_ULTOSC( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod1 , timeperiod2 , timeperiod3 , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ULTOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def VAR( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """ VAR(real[, timeperiod=?, nbdev=?])

    Variance (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdev: 1
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_VAR( length - 1 , length - 1 , real_data , timeperiod , nbdev , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_VAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WCLPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ WCLPRICE(high, low, close)

    Weighted Close Price (Price Transform)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_WCLPRICE( length - 1 , length - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WCLPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WILLR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ WILLR(high, low, close[, timeperiod=?])

    Williams' %R (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(high) != np.NPY_DOUBLE:
        raise Exception("high is not double")
    if high.ndim != 1:
        raise Exception("high has wrong dimensions")
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    if PyArray_TYPE(low) != np.NPY_DOUBLE:
        raise Exception("low is not double")
    if low.ndim != 1:
        raise Exception("low has wrong dimensions")
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    if PyArray_TYPE(close) != np.NPY_DOUBLE:
        raise Exception("close is not double")
    if close.ndim != 1:
        raise Exception("close has wrong dimensions")
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    if length != low.shape[0]:
        raise Exception("input lengths are different")
    if length != close.shape[0]:
        raise Exception("input lengths are different")
    outreal = NaN
    retCode = lib.TA_WILLR( length - 1 , length - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WILLR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """ WMA(real[, timeperiod=?])

    Weighted Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        double val
        int begidx, endidx, lookback
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("real is not double")
    if real.ndim != 1:
        raise Exception("real has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_WMA( length - 1 , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WMA", retCode)
    return outreal 

__all__ = ["ACOS","AD","ADD","ADOSC","ADX","ADXR","APO","AROON","AROONOSC","ASIN","ATAN","ATR","AVGPRICE","BBANDS","BETA","BOP","CCI","CDL2CROWS","CDL3BLACKCROWS","CDL3INSIDE","CDL3LINESTRIKE","CDL3OUTSIDE","CDL3STARSINSOUTH","CDL3WHITESOLDIERS","CDLABANDONEDBABY","CDLADVANCEBLOCK","CDLBELTHOLD","CDLBREAKAWAY","CDLCLOSINGMARUBOZU","CDLCONCEALBABYSWALL","CDLCOUNTERATTACK","CDLDARKCLOUDCOVER","CDLDOJI","CDLDOJISTAR","CDLDRAGONFLYDOJI","CDLENGULFING","CDLEVENINGDOJISTAR","CDLEVENINGSTAR","CDLGAPSIDESIDEWHITE","CDLGRAVESTONEDOJI","CDLHAMMER","CDLHANGINGMAN","CDLHARAMI","CDLHARAMICROSS","CDLHIGHWAVE","CDLHIKKAKE","CDLHIKKAKEMOD","CDLHOMINGPIGEON","CDLIDENTICAL3CROWS","CDLINNECK","CDLINVERTEDHAMMER","CDLKICKING","CDLKICKINGBYLENGTH","CDLLADDERBOTTOM","CDLLONGLEGGEDDOJI","CDLLONGLINE","CDLMARUBOZU","CDLMATCHINGLOW","CDLMATHOLD","CDLMORNINGDOJISTAR","CDLMORNINGSTAR","CDLONNECK","CDLPIERCING","CDLRICKSHAWMAN","CDLRISEFALL3METHODS","CDLSEPARATINGLINES","CDLSHOOTINGSTAR","CDLSHORTLINE","CDLSPINNINGTOP","CDLSTALLEDPATTERN","CDLSTICKSANDWICH","CDLTAKURI","CDLTASUKIGAP","CDLTHRUSTING","CDLTRISTAR","CDLUNIQUE3RIVER","CDLUPSIDEGAP2CROWS","CDLXSIDEGAP3METHODS","CEIL","CMO","CORREL","COS","COSH","DEMA","DIV","DX","EMA","EXP","FLOOR","HT_DCPERIOD","HT_DCPHASE","HT_PHASOR","HT_SINE","HT_TRENDLINE","HT_TRENDMODE","KAMA","LINEARREG","LINEARREG_ANGLE","LINEARREG_INTERCEPT","LINEARREG_SLOPE","LN","LOG10","MA","MACD","MACDEXT","MACDFIX","MAMA","MAVP","MAX","MAXINDEX","MEDPRICE","MFI","MIDPOINT","MIDPRICE","MIN","MININDEX","MINMAX","MINMAXINDEX","MINUS_DI","MINUS_DM","MOM","MULT","NATR","OBV","PLUS_DI","PLUS_DM","PPO","ROC","ROCP","ROCR","ROCR100","RSI","SAR","SAREXT","SIN","SINH","SMA","SQRT","STDDEV","STOCH","STOCHF","STOCHRSI","SUB","SUM","T3","TAN","TANH","TEMA","TRANGE","TRIMA","TRIX","TSF","TYPPRICE","ULTOSC","VAR","WCLPRICE","WILLR","WMA"]
