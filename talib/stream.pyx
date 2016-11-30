cimport numpy as np

try:
    import pandas
    __PANDAS_SERIES = pandas.Series
except ImportError:
    __PANDAS_SERIES = None

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

def ACOS( real not None ):
    """ ACOS(real)

    Vector Trigonometric ACos (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _ACOS(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ACOS( np.ndarray real ):
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

def AD( high not None, low not None, close not None, volume not None ):
    """ AD(high, low, close, volume)

    Chaikin A/D Line (Volume Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Outputs:
        real
    """
    return _AD(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        volume.values if isinstance(volume, __PANDAS_SERIES) else volume
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _AD( np.ndarray high, np.ndarray low, np.ndarray close, np.ndarray volume ):
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

def ADD( real0 not None, real1 not None ):
    """ ADD(real0, real1)

    Vector Arithmetic Add (Math Operators)

    Inputs:
        real0: (np.ndarray or pd.Series)
        real1: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _ADD(
        real0.values if isinstance(real0, __PANDAS_SERIES) else real0,
        real1.values if isinstance(real1, __PANDAS_SERIES) else real1
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ADD( np.ndarray real0, np.ndarray real1 ):
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

def ADOSC( high not None, low not None, close not None, volume not None, fastperiod=3, slowperiod=10 ):
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
    return _ADOSC(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        volume.values if isinstance(volume, __PANDAS_SERIES) else volume,
        fastperiod,
        slowperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ADOSC( np.ndarray high, np.ndarray low, np.ndarray close, np.ndarray volume, int fastperiod=3, int slowperiod=10 ):
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

def ADX( high not None, low not None, close not None, timeperiod=14 ):
    """ ADX(high, low, close[, timeperiod=?])

    Average Directional Movement Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _ADX(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ADX( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def ADXR( high not None, low not None, close not None, timeperiod=14 ):
    """ ADXR(high, low, close[, timeperiod=?])

    Average Directional Movement Index Rating (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _ADXR(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ADXR( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def APO( real not None, fastperiod=12, slowperiod=26, matype=0 ):
    """ APO(real[, fastperiod=?, slowperiod=?, matype=?])

    Absolute Price Oscillator (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    return _APO(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        fastperiod,
        slowperiod,
        matype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _APO( np.ndarray real, int fastperiod=12, int slowperiod=26, int matype=0 ):
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

def AROON( high not None, low not None, timeperiod=14 ):
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
    return _AROON(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _AROON( np.ndarray high, np.ndarray low, int timeperiod=14 ):
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

def AROONOSC( high not None, low not None, timeperiod=14 ):
    """ AROONOSC(high, low[, timeperiod=?])

    Aroon Oscillator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _AROONOSC(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _AROONOSC( np.ndarray high, np.ndarray low, int timeperiod=14 ):
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

def ASIN( real not None ):
    """ ASIN(real)

    Vector Trigonometric ASin (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _ASIN(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ASIN( np.ndarray real ):
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

def ATAN( real not None ):
    """ ATAN(real)

    Vector Trigonometric ATan (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _ATAN(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ATAN( np.ndarray real ):
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

def ATR( high not None, low not None, close not None, timeperiod=14 ):
    """ ATR(high, low, close[, timeperiod=?])

    Average True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _ATR(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ATR( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def AVGPRICE( open not None, high not None, low not None, close not None ):
    """ AVGPRICE(open, high, low, close)

    Average Price (Price Transform)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        real
    """
    return _AVGPRICE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _AVGPRICE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def BBANDS( real not None, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0 ):
    """ BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])

    Bollinger Bands (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
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
    return _BBANDS(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod,
        nbdevup,
        nbdevdn,
        matype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _BBANDS( np.ndarray real, int timeperiod=5, double nbdevup=2, double nbdevdn=2, int matype=0 ):
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

def BETA( real0 not None, real1 not None, timeperiod=5 ):
    """ BETA(real0, real1[, timeperiod=?])

    Beta (Statistic Functions)

    Inputs:
        real0: (np.ndarray or pd.Series)
        real1: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 5
    Outputs:
        real
    """
    return _BETA(
        real0.values if isinstance(real0, __PANDAS_SERIES) else real0,
        real1.values if isinstance(real1, __PANDAS_SERIES) else real1,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _BETA( np.ndarray real0, np.ndarray real1, int timeperiod=5 ):
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

def BOP( open not None, high not None, low not None, close not None ):
    """ BOP(open, high, low, close)

    Balance Of Power (Momentum Indicators)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        real
    """
    return _BOP(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _BOP( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CCI( high not None, low not None, close not None, timeperiod=14 ):
    """ CCI(high, low, close[, timeperiod=?])

    Commodity Channel Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _CCI(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CCI( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def CDL2CROWS( open not None, high not None, low not None, close not None ):
    """ CDL2CROWS(open, high, low, close)

    Two Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDL2CROWS(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDL2CROWS( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDL3BLACKCROWS( open not None, high not None, low not None, close not None ):
    """ CDL3BLACKCROWS(open, high, low, close)

    Three Black Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDL3BLACKCROWS(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDL3BLACKCROWS( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDL3INSIDE( open not None, high not None, low not None, close not None ):
    """ CDL3INSIDE(open, high, low, close)

    Three Inside Up/Down (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDL3INSIDE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDL3INSIDE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDL3LINESTRIKE( open not None, high not None, low not None, close not None ):
    """ CDL3LINESTRIKE(open, high, low, close)

    Three-Line Strike  (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDL3LINESTRIKE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDL3LINESTRIKE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDL3OUTSIDE( open not None, high not None, low not None, close not None ):
    """ CDL3OUTSIDE(open, high, low, close)

    Three Outside Up/Down (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDL3OUTSIDE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDL3OUTSIDE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDL3STARSINSOUTH( open not None, high not None, low not None, close not None ):
    """ CDL3STARSINSOUTH(open, high, low, close)

    Three Stars In The South (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDL3STARSINSOUTH(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDL3STARSINSOUTH( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDL3WHITESOLDIERS( open not None, high not None, low not None, close not None ):
    """ CDL3WHITESOLDIERS(open, high, low, close)

    Three Advancing White Soldiers (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDL3WHITESOLDIERS(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDL3WHITESOLDIERS( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLABANDONEDBABY( open not None, high not None, low not None, close not None, penetration=0.3 ):
    """ CDLABANDONEDBABY(open, high, low, close[, penetration=?])

    Abandoned Baby (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLABANDONEDBABY(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        penetration
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLABANDONEDBABY( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close, double penetration=0.3 ):
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

def CDLADVANCEBLOCK( open not None, high not None, low not None, close not None ):
    """ CDLADVANCEBLOCK(open, high, low, close)

    Advance Block (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLADVANCEBLOCK(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLADVANCEBLOCK( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLBELTHOLD( open not None, high not None, low not None, close not None ):
    """ CDLBELTHOLD(open, high, low, close)

    Belt-hold (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLBELTHOLD(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLBELTHOLD( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLBREAKAWAY( open not None, high not None, low not None, close not None ):
    """ CDLBREAKAWAY(open, high, low, close)

    Breakaway (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLBREAKAWAY(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLBREAKAWAY( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLCLOSINGMARUBOZU( open not None, high not None, low not None, close not None ):
    """ CDLCLOSINGMARUBOZU(open, high, low, close)

    Closing Marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLCLOSINGMARUBOZU(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLCLOSINGMARUBOZU( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLCONCEALBABYSWALL( open not None, high not None, low not None, close not None ):
    """ CDLCONCEALBABYSWALL(open, high, low, close)

    Concealing Baby Swallow (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLCONCEALBABYSWALL(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLCONCEALBABYSWALL( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLCOUNTERATTACK( open not None, high not None, low not None, close not None ):
    """ CDLCOUNTERATTACK(open, high, low, close)

    Counterattack (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLCOUNTERATTACK(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLCOUNTERATTACK( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLDARKCLOUDCOVER( open not None, high not None, low not None, close not None, penetration=0.5 ):
    """ CDLDARKCLOUDCOVER(open, high, low, close[, penetration=?])

    Dark Cloud Cover (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.5
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLDARKCLOUDCOVER(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        penetration
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLDARKCLOUDCOVER( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close, double penetration=0.5 ):
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

def CDLDOJI( open not None, high not None, low not None, close not None ):
    """ CDLDOJI(open, high, low, close)

    Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLDOJI(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLDOJI( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLDOJISTAR( open not None, high not None, low not None, close not None ):
    """ CDLDOJISTAR(open, high, low, close)

    Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLDOJISTAR(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLDOJISTAR( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLDRAGONFLYDOJI( open not None, high not None, low not None, close not None ):
    """ CDLDRAGONFLYDOJI(open, high, low, close)

    Dragonfly Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLDRAGONFLYDOJI(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLDRAGONFLYDOJI( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLENGULFING( open not None, high not None, low not None, close not None ):
    """ CDLENGULFING(open, high, low, close)

    Engulfing Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLENGULFING(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLENGULFING( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLEVENINGDOJISTAR( open not None, high not None, low not None, close not None, penetration=0.3 ):
    """ CDLEVENINGDOJISTAR(open, high, low, close[, penetration=?])

    Evening Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLEVENINGDOJISTAR(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        penetration
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLEVENINGDOJISTAR( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close, double penetration=0.3 ):
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

def CDLEVENINGSTAR( open not None, high not None, low not None, close not None, penetration=0.3 ):
    """ CDLEVENINGSTAR(open, high, low, close[, penetration=?])

    Evening Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLEVENINGSTAR(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        penetration
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLEVENINGSTAR( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close, double penetration=0.3 ):
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

def CDLGAPSIDESIDEWHITE( open not None, high not None, low not None, close not None ):
    """ CDLGAPSIDESIDEWHITE(open, high, low, close)

    Up/Down-gap side-by-side white lines (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLGAPSIDESIDEWHITE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLGAPSIDESIDEWHITE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLGRAVESTONEDOJI( open not None, high not None, low not None, close not None ):
    """ CDLGRAVESTONEDOJI(open, high, low, close)

    Gravestone Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLGRAVESTONEDOJI(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLGRAVESTONEDOJI( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLHAMMER( open not None, high not None, low not None, close not None ):
    """ CDLHAMMER(open, high, low, close)

    Hammer (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLHAMMER(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLHAMMER( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLHANGINGMAN( open not None, high not None, low not None, close not None ):
    """ CDLHANGINGMAN(open, high, low, close)

    Hanging Man (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLHANGINGMAN(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLHANGINGMAN( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLHARAMI( open not None, high not None, low not None, close not None ):
    """ CDLHARAMI(open, high, low, close)

    Harami Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLHARAMI(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLHARAMI( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLHARAMICROSS( open not None, high not None, low not None, close not None ):
    """ CDLHARAMICROSS(open, high, low, close)

    Harami Cross Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLHARAMICROSS(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLHARAMICROSS( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLHIGHWAVE( open not None, high not None, low not None, close not None ):
    """ CDLHIGHWAVE(open, high, low, close)

    High-Wave Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLHIGHWAVE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLHIGHWAVE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLHIKKAKE( open not None, high not None, low not None, close not None ):
    """ CDLHIKKAKE(open, high, low, close)

    Hikkake Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLHIKKAKE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLHIKKAKE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLHIKKAKEMOD( open not None, high not None, low not None, close not None ):
    """ CDLHIKKAKEMOD(open, high, low, close)

    Modified Hikkake Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLHIKKAKEMOD(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLHIKKAKEMOD( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLHOMINGPIGEON( open not None, high not None, low not None, close not None ):
    """ CDLHOMINGPIGEON(open, high, low, close)

    Homing Pigeon (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLHOMINGPIGEON(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLHOMINGPIGEON( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLIDENTICAL3CROWS( open not None, high not None, low not None, close not None ):
    """ CDLIDENTICAL3CROWS(open, high, low, close)

    Identical Three Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLIDENTICAL3CROWS(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLIDENTICAL3CROWS( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLINNECK( open not None, high not None, low not None, close not None ):
    """ CDLINNECK(open, high, low, close)

    In-Neck Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLINNECK(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLINNECK( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLINVERTEDHAMMER( open not None, high not None, low not None, close not None ):
    """ CDLINVERTEDHAMMER(open, high, low, close)

    Inverted Hammer (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLINVERTEDHAMMER(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLINVERTEDHAMMER( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLKICKING( open not None, high not None, low not None, close not None ):
    """ CDLKICKING(open, high, low, close)

    Kicking (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLKICKING(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLKICKING( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLKICKINGBYLENGTH( open not None, high not None, low not None, close not None ):
    """ CDLKICKINGBYLENGTH(open, high, low, close)

    Kicking - bull/bear determined by the longer marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLKICKINGBYLENGTH(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLKICKINGBYLENGTH( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLLADDERBOTTOM( open not None, high not None, low not None, close not None ):
    """ CDLLADDERBOTTOM(open, high, low, close)

    Ladder Bottom (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLLADDERBOTTOM(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLLADDERBOTTOM( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLLONGLEGGEDDOJI( open not None, high not None, low not None, close not None ):
    """ CDLLONGLEGGEDDOJI(open, high, low, close)

    Long Legged Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLLONGLEGGEDDOJI(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLLONGLEGGEDDOJI( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLLONGLINE( open not None, high not None, low not None, close not None ):
    """ CDLLONGLINE(open, high, low, close)

    Long Line Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLLONGLINE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLLONGLINE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLMARUBOZU( open not None, high not None, low not None, close not None ):
    """ CDLMARUBOZU(open, high, low, close)

    Marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLMARUBOZU(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLMARUBOZU( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLMATCHINGLOW( open not None, high not None, low not None, close not None ):
    """ CDLMATCHINGLOW(open, high, low, close)

    Matching Low (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLMATCHINGLOW(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLMATCHINGLOW( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLMATHOLD( open not None, high not None, low not None, close not None, penetration=0.5 ):
    """ CDLMATHOLD(open, high, low, close[, penetration=?])

    Mat Hold (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.5
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLMATHOLD(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        penetration
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLMATHOLD( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close, double penetration=0.5 ):
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

def CDLMORNINGDOJISTAR( open not None, high not None, low not None, close not None, penetration=0.3 ):
    """ CDLMORNINGDOJISTAR(open, high, low, close[, penetration=?])

    Morning Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLMORNINGDOJISTAR(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        penetration
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLMORNINGDOJISTAR( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close, double penetration=0.3 ):
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

def CDLMORNINGSTAR( open not None, high not None, low not None, close not None, penetration=0.3 ):
    """ CDLMORNINGSTAR(open, high, low, close[, penetration=?])

    Morning Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLMORNINGSTAR(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        penetration
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLMORNINGSTAR( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close, double penetration=0.3 ):
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

def CDLONNECK( open not None, high not None, low not None, close not None ):
    """ CDLONNECK(open, high, low, close)

    On-Neck Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLONNECK(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLONNECK( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLPIERCING( open not None, high not None, low not None, close not None ):
    """ CDLPIERCING(open, high, low, close)

    Piercing Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLPIERCING(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLPIERCING( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLRICKSHAWMAN( open not None, high not None, low not None, close not None ):
    """ CDLRICKSHAWMAN(open, high, low, close)

    Rickshaw Man (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLRICKSHAWMAN(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLRICKSHAWMAN( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLRISEFALL3METHODS( open not None, high not None, low not None, close not None ):
    """ CDLRISEFALL3METHODS(open, high, low, close)

    Rising/Falling Three Methods (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLRISEFALL3METHODS(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLRISEFALL3METHODS( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLSEPARATINGLINES( open not None, high not None, low not None, close not None ):
    """ CDLSEPARATINGLINES(open, high, low, close)

    Separating Lines (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLSEPARATINGLINES(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLSEPARATINGLINES( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLSHOOTINGSTAR( open not None, high not None, low not None, close not None ):
    """ CDLSHOOTINGSTAR(open, high, low, close)

    Shooting Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLSHOOTINGSTAR(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLSHOOTINGSTAR( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLSHORTLINE( open not None, high not None, low not None, close not None ):
    """ CDLSHORTLINE(open, high, low, close)

    Short Line Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLSHORTLINE(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLSHORTLINE( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLSPINNINGTOP( open not None, high not None, low not None, close not None ):
    """ CDLSPINNINGTOP(open, high, low, close)

    Spinning Top (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLSPINNINGTOP(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLSPINNINGTOP( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLSTALLEDPATTERN( open not None, high not None, low not None, close not None ):
    """ CDLSTALLEDPATTERN(open, high, low, close)

    Stalled Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLSTALLEDPATTERN(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLSTALLEDPATTERN( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLSTICKSANDWICH( open not None, high not None, low not None, close not None ):
    """ CDLSTICKSANDWICH(open, high, low, close)

    Stick Sandwich (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLSTICKSANDWICH(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLSTICKSANDWICH( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLTAKURI( open not None, high not None, low not None, close not None ):
    """ CDLTAKURI(open, high, low, close)

    Takuri (Dragonfly Doji with very long lower shadow) (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLTAKURI(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLTAKURI( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLTASUKIGAP( open not None, high not None, low not None, close not None ):
    """ CDLTASUKIGAP(open, high, low, close)

    Tasuki Gap (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLTASUKIGAP(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLTASUKIGAP( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLTHRUSTING( open not None, high not None, low not None, close not None ):
    """ CDLTHRUSTING(open, high, low, close)

    Thrusting Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLTHRUSTING(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLTHRUSTING( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLTRISTAR( open not None, high not None, low not None, close not None ):
    """ CDLTRISTAR(open, high, low, close)

    Tristar Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLTRISTAR(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLTRISTAR( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLUNIQUE3RIVER( open not None, high not None, low not None, close not None ):
    """ CDLUNIQUE3RIVER(open, high, low, close)

    Unique 3 River (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLUNIQUE3RIVER(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLUNIQUE3RIVER( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLUPSIDEGAP2CROWS( open not None, high not None, low not None, close not None ):
    """ CDLUPSIDEGAP2CROWS(open, high, low, close)

    Upside Gap Two Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLUPSIDEGAP2CROWS(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLUPSIDEGAP2CROWS( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CDLXSIDEGAP3METHODS( open not None, high not None, low not None, close not None ):
    """ CDLXSIDEGAP3METHODS(open, high, low, close)

    Upside/Downside Gap Three Methods (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _CDLXSIDEGAP3METHODS(
        open.values if isinstance(open, __PANDAS_SERIES) else open,
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CDLXSIDEGAP3METHODS( np.ndarray open, np.ndarray high, np.ndarray low, np.ndarray close ):
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

def CEIL( real not None ):
    """ CEIL(real)

    Vector Ceil (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _CEIL(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CEIL( np.ndarray real ):
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

def CMO( real not None, timeperiod=14 ):
    """ CMO(real[, timeperiod=?])

    Chande Momentum Oscillator (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _CMO(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CMO( np.ndarray real, int timeperiod=14 ):
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

def CORREL( real0 not None, real1 not None, timeperiod=30 ):
    """ CORREL(real0, real1[, timeperiod=?])

    Pearson's Correlation Coefficient (r) (Statistic Functions)

    Inputs:
        real0: (np.ndarray or pd.Series)
        real1: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _CORREL(
        real0.values if isinstance(real0, __PANDAS_SERIES) else real0,
        real1.values if isinstance(real1, __PANDAS_SERIES) else real1,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _CORREL( np.ndarray real0, np.ndarray real1, int timeperiod=30 ):
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

def COS( real not None ):
    """ COS(real)

    Vector Trigonometric Cos (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _COS(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _COS( np.ndarray real ):
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

def COSH( real not None ):
    """ COSH(real)

    Vector Trigonometric Cosh (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _COSH(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _COSH( np.ndarray real ):
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

def DEMA( real not None, timeperiod=30 ):
    """ DEMA(real[, timeperiod=?])

    Double Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _DEMA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _DEMA( np.ndarray real, int timeperiod=30 ):
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

def DIV( real0 not None, real1 not None ):
    """ DIV(real0, real1)

    Vector Arithmetic Div (Math Operators)

    Inputs:
        real0: (np.ndarray or pd.Series)
        real1: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _DIV(
        real0.values if isinstance(real0, __PANDAS_SERIES) else real0,
        real1.values if isinstance(real1, __PANDAS_SERIES) else real1
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _DIV( np.ndarray real0, np.ndarray real1 ):
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

def DX( high not None, low not None, close not None, timeperiod=14 ):
    """ DX(high, low, close[, timeperiod=?])

    Directional Movement Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _DX(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _DX( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def EMA( real not None, timeperiod=30 ):
    """ EMA(real[, timeperiod=?])

    Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _EMA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _EMA( np.ndarray real, int timeperiod=30 ):
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

def EXP( real not None ):
    """ EXP(real)

    Vector Arithmetic Exp (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _EXP(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _EXP( np.ndarray real ):
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

def FLOOR( real not None ):
    """ FLOOR(real)

    Vector Floor (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _FLOOR(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _FLOOR( np.ndarray real ):
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

def HT_DCPERIOD( real not None ):
    """ HT_DCPERIOD(real)

    Hilbert Transform - Dominant Cycle Period (Cycle Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _HT_DCPERIOD(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _HT_DCPERIOD( np.ndarray real ):
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

def HT_DCPHASE( real not None ):
    """ HT_DCPHASE(real)

    Hilbert Transform - Dominant Cycle Phase (Cycle Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _HT_DCPHASE(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _HT_DCPHASE( np.ndarray real ):
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

def HT_PHASOR( real not None ):
    """ HT_PHASOR(real)

    Hilbert Transform - Phasor Components (Cycle Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        inphase
        quadrature
    """
    return _HT_PHASOR(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _HT_PHASOR( np.ndarray real ):
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

def HT_SINE( real not None ):
    """ HT_SINE(real)

    Hilbert Transform - SineWave (Cycle Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        sine
        leadsine
    """
    return _HT_SINE(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _HT_SINE( np.ndarray real ):
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

def HT_TRENDLINE( real not None ):
    """ HT_TRENDLINE(real)

    Hilbert Transform - Instantaneous Trendline (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _HT_TRENDLINE(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _HT_TRENDLINE( np.ndarray real ):
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

def HT_TRENDMODE( real not None ):
    """ HT_TRENDMODE(real)

    Hilbert Transform - Trend vs Cycle Mode (Cycle Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _HT_TRENDMODE(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _HT_TRENDMODE( np.ndarray real ):
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

def KAMA( real not None, timeperiod=30 ):
    """ KAMA(real[, timeperiod=?])

    Kaufman Adaptive Moving Average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _KAMA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _KAMA( np.ndarray real, int timeperiod=30 ):
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

def LINEARREG( real not None, timeperiod=14 ):
    """ LINEARREG(real[, timeperiod=?])

    Linear Regression (Statistic Functions)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _LINEARREG(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _LINEARREG( np.ndarray real, int timeperiod=14 ):
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

def LINEARREG_ANGLE( real not None, timeperiod=14 ):
    """ LINEARREG_ANGLE(real[, timeperiod=?])

    Linear Regression Angle (Statistic Functions)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _LINEARREG_ANGLE(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _LINEARREG_ANGLE( np.ndarray real, int timeperiod=14 ):
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

def LINEARREG_INTERCEPT( real not None, timeperiod=14 ):
    """ LINEARREG_INTERCEPT(real[, timeperiod=?])

    Linear Regression Intercept (Statistic Functions)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _LINEARREG_INTERCEPT(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _LINEARREG_INTERCEPT( np.ndarray real, int timeperiod=14 ):
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

def LINEARREG_SLOPE( real not None, timeperiod=14 ):
    """ LINEARREG_SLOPE(real[, timeperiod=?])

    Linear Regression Slope (Statistic Functions)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _LINEARREG_SLOPE(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _LINEARREG_SLOPE( np.ndarray real, int timeperiod=14 ):
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

def LN( real not None ):
    """ LN(real)

    Vector Log Natural (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _LN(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _LN( np.ndarray real ):
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

def LOG10( real not None ):
    """ LOG10(real)

    Vector Log10 (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _LOG10(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _LOG10( np.ndarray real ):
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

def MA( real not None, timeperiod=30, matype=0 ):
    """ MA(real[, timeperiod=?, matype=?])

    Moving average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    return _MA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod,
        matype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MA( np.ndarray real, int timeperiod=30, int matype=0 ):
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

def MACD( real not None, fastperiod=12, slowperiod=26, signalperiod=9 ):
    """ MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])

    Moving Average Convergence/Divergence (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        signalperiod: 9
    Outputs:
        macd
        macdsignal
        macdhist
    """
    return _MACD(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        fastperiod,
        slowperiod,
        signalperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MACD( np.ndarray real, int fastperiod=12, int slowperiod=26, int signalperiod=9 ):
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

def MACDEXT( real not None, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0 ):
    """ MACDEXT(real[, fastperiod=?, fastmatype=?, slowperiod=?, slowmatype=?, signalperiod=?, signalmatype=?])

    MACD with controllable MA type (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
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
    return _MACDEXT(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        fastperiod,
        fastmatype,
        slowperiod,
        slowmatype,
        signalperiod,
        signalmatype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MACDEXT( np.ndarray real, int fastperiod=12, int fastmatype=0, int slowperiod=26, int slowmatype=0, int signalperiod=9, int signalmatype=0 ):
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

def MACDFIX( real not None, signalperiod=9 ):
    """ MACDFIX(real[, signalperiod=?])

    Moving Average Convergence/Divergence Fix 12/26 (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        signalperiod: 9
    Outputs:
        macd
        macdsignal
        macdhist
    """
    return _MACDFIX(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        signalperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MACDFIX( np.ndarray real, int signalperiod=9 ):
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

def MAMA( real not None, fastlimit=0.5, slowlimit=0.05 ):
    """ MAMA(real[, fastlimit=?, slowlimit=?])

    MESA Adaptive Moving Average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        fastlimit: 0.5
        slowlimit: 0.05
    Outputs:
        mama
        fama
    """
    return _MAMA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        fastlimit,
        slowlimit
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MAMA( np.ndarray real, double fastlimit=0.5, double slowlimit=0.05 ):
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

def MAVP( real not None, periods not None, minperiod=2, maxperiod=30, matype=0 ):
    """ MAVP(real, periods[, minperiod=?, maxperiod=?, matype=?])

    Moving average with variable period (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
        periods: (np.ndarray or pd.Series)
    Parameters:
        minperiod: 2
        maxperiod: 30
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    return _MAVP(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        periods.values if isinstance(periods, __PANDAS_SERIES) else periods,
        minperiod,
        maxperiod,
        matype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MAVP( np.ndarray real, np.ndarray periods, int minperiod=2, int maxperiod=30, int matype=0 ):
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

def MAX( real not None, timeperiod=30 ):
    """ MAX(real[, timeperiod=?])

    Highest value over a specified period (Math Operators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _MAX(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MAX( np.ndarray real, int timeperiod=30 ):
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

def MAXINDEX( real not None, timeperiod=30 ):
    """ MAXINDEX(real[, timeperiod=?])

    Index of highest value over a specified period (Math Operators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _MAXINDEX(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MAXINDEX( np.ndarray real, int timeperiod=30 ):
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

def MEDPRICE( high not None, low not None ):
    """ MEDPRICE(high, low)

    Median Price (Price Transform)

    Inputs:
        prices: ['high', 'low']
    Outputs:
        real
    """
    return _MEDPRICE(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MEDPRICE( np.ndarray high, np.ndarray low ):
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

def MFI( high not None, low not None, close not None, volume not None, timeperiod=14 ):
    """ MFI(high, low, close, volume[, timeperiod=?])

    Money Flow Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _MFI(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        volume.values if isinstance(volume, __PANDAS_SERIES) else volume,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MFI( np.ndarray high, np.ndarray low, np.ndarray close, np.ndarray volume, int timeperiod=14 ):
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

def MIDPOINT( real not None, timeperiod=14 ):
    """ MIDPOINT(real[, timeperiod=?])

    MidPoint over period (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _MIDPOINT(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MIDPOINT( np.ndarray real, int timeperiod=14 ):
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

def MIDPRICE( high not None, low not None, timeperiod=14 ):
    """ MIDPRICE(high, low[, timeperiod=?])

    Midpoint Price over period (Overlap Studies)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _MIDPRICE(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MIDPRICE( np.ndarray high, np.ndarray low, int timeperiod=14 ):
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

def MIN( real not None, timeperiod=30 ):
    """ MIN(real[, timeperiod=?])

    Lowest value over a specified period (Math Operators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _MIN(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MIN( np.ndarray real, int timeperiod=30 ):
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

def MININDEX( real not None, timeperiod=30 ):
    """ MININDEX(real[, timeperiod=?])

    Index of lowest value over a specified period (Math Operators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        integer (values are -100, 0 or 100)
    """
    return _MININDEX(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MININDEX( np.ndarray real, int timeperiod=30 ):
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

def MINMAX( real not None, timeperiod=30 ):
    """ MINMAX(real[, timeperiod=?])

    Lowest and highest values over a specified period (Math Operators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        min
        max
    """
    return _MINMAX(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MINMAX( np.ndarray real, int timeperiod=30 ):
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

def MINMAXINDEX( real not None, timeperiod=30 ):
    """ MINMAXINDEX(real[, timeperiod=?])

    Indexes of lowest and highest values over a specified period (Math Operators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        minidx
        maxidx
    """
    return _MINMAXINDEX(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MINMAXINDEX( np.ndarray real, int timeperiod=30 ):
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

def MINUS_DI( high not None, low not None, close not None, timeperiod=14 ):
    """ MINUS_DI(high, low, close[, timeperiod=?])

    Minus Directional Indicator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _MINUS_DI(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MINUS_DI( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def MINUS_DM( high not None, low not None, timeperiod=14 ):
    """ MINUS_DM(high, low[, timeperiod=?])

    Minus Directional Movement (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _MINUS_DM(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MINUS_DM( np.ndarray high, np.ndarray low, int timeperiod=14 ):
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

def MOM( real not None, timeperiod=10 ):
    """ MOM(real[, timeperiod=?])

    Momentum (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    return _MOM(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MOM( np.ndarray real, int timeperiod=10 ):
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

def MULT( real0 not None, real1 not None ):
    """ MULT(real0, real1)

    Vector Arithmetic Mult (Math Operators)

    Inputs:
        real0: (np.ndarray or pd.Series)
        real1: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _MULT(
        real0.values if isinstance(real0, __PANDAS_SERIES) else real0,
        real1.values if isinstance(real1, __PANDAS_SERIES) else real1
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _MULT( np.ndarray real0, np.ndarray real1 ):
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

def NATR( high not None, low not None, close not None, timeperiod=14 ):
    """ NATR(high, low, close[, timeperiod=?])

    Normalized Average True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _NATR(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _NATR( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def OBV( real not None, volume not None ):
    """ OBV(real, volume)

    On Balance Volume (Volume Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
        prices: ['volume']
    Outputs:
        real
    """
    return _OBV(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        volume.values if isinstance(volume, __PANDAS_SERIES) else volume
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _OBV( np.ndarray real, np.ndarray volume ):
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

def PLUS_DI( high not None, low not None, close not None, timeperiod=14 ):
    """ PLUS_DI(high, low, close[, timeperiod=?])

    Plus Directional Indicator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _PLUS_DI(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _PLUS_DI( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def PLUS_DM( high not None, low not None, timeperiod=14 ):
    """ PLUS_DM(high, low[, timeperiod=?])

    Plus Directional Movement (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _PLUS_DM(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _PLUS_DM( np.ndarray high, np.ndarray low, int timeperiod=14 ):
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

def PPO( real not None, fastperiod=12, slowperiod=26, matype=0 ):
    """ PPO(real[, fastperiod=?, slowperiod=?, matype=?])

    Percentage Price Oscillator (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    return _PPO(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        fastperiod,
        slowperiod,
        matype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _PPO( np.ndarray real, int fastperiod=12, int slowperiod=26, int matype=0 ):
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

def ROC( real not None, timeperiod=10 ):
    """ ROC(real[, timeperiod=?])

    Rate of change : ((real/prevPrice)-1)*100 (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    return _ROC(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ROC( np.ndarray real, int timeperiod=10 ):
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

def ROCP( real not None, timeperiod=10 ):
    """ ROCP(real[, timeperiod=?])

    Rate of change Percentage: (real-prevPrice)/prevPrice (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    return _ROCP(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ROCP( np.ndarray real, int timeperiod=10 ):
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

def ROCR( real not None, timeperiod=10 ):
    """ ROCR(real[, timeperiod=?])

    Rate of change ratio: (real/prevPrice) (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    return _ROCR(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ROCR( np.ndarray real, int timeperiod=10 ):
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

def ROCR100( real not None, timeperiod=10 ):
    """ ROCR100(real[, timeperiod=?])

    Rate of change ratio 100 scale: (real/prevPrice)*100 (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    return _ROCR100(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ROCR100( np.ndarray real, int timeperiod=10 ):
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

def RSI( real not None, timeperiod=14 ):
    """ RSI(real[, timeperiod=?])

    Relative Strength Index (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _RSI(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _RSI( np.ndarray real, int timeperiod=14 ):
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

def SAR( high not None, low not None, acceleration=0.02, maximum=0.2 ):
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
    return _SAR(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        acceleration,
        maximum
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _SAR( np.ndarray high, np.ndarray low, double acceleration=0.02, double maximum=0.2 ):
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

def SAREXT( high not None, low not None, startvalue=0.0, offsetonreverse=0.0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2 ):
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
    return _SAREXT(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        startvalue,
        offsetonreverse,
        accelerationinitlong,
        accelerationlong,
        accelerationmaxlong,
        accelerationinitshort,
        accelerationshort,
        accelerationmaxshort
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _SAREXT( np.ndarray high, np.ndarray low, double startvalue=0, double offsetonreverse=0, double accelerationinitlong=0.02, double accelerationlong=0.02, double accelerationmaxlong=0.2, double accelerationinitshort=0.02, double accelerationshort=0.02, double accelerationmaxshort=0.2 ):
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

def SIN( real not None ):
    """ SIN(real)

    Vector Trigonometric Sin (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _SIN(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _SIN( np.ndarray real ):
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

def SINH( real not None ):
    """ SINH(real)

    Vector Trigonometric Sinh (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _SINH(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _SINH( np.ndarray real ):
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

def SMA( real not None, timeperiod=30 ):
    """ SMA(real[, timeperiod=?])

    Simple Moving Average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _SMA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _SMA( np.ndarray real, int timeperiod=30 ):
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

def SQRT( real not None ):
    """ SQRT(real)

    Vector Square Root (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _SQRT(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _SQRT( np.ndarray real ):
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

def STDDEV( real not None, timeperiod=5, nbdev=1.0 ):
    """ STDDEV(real[, timeperiod=?, nbdev=?])

    Standard Deviation (Statistic Functions)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 5
        nbdev: 1
    Outputs:
        real
    """
    return _STDDEV(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod,
        nbdev
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _STDDEV( np.ndarray real, int timeperiod=5, double nbdev=1 ):
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

def STOCH( high not None, low not None, close not None, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0 ):
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
    return _STOCH(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        fastk_period,
        slowk_period,
        slowk_matype,
        slowd_period,
        slowd_matype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _STOCH( np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period=5, int slowk_period=3, int slowk_matype=0, int slowd_period=3, int slowd_matype=0 ):
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

def STOCHF( high not None, low not None, close not None, fastk_period=5, fastd_period=3, fastd_matype=0 ):
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
    return _STOCHF(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        fastk_period,
        fastd_period,
        fastd_matype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _STOCHF( np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period=5, int fastd_period=3, int fastd_matype=0 ):
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

def STOCHRSI( real not None, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0 ):
    """ STOCHRSI(real[, timeperiod=?, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Relative Strength Index (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
        fastk_period: 5
        fastd_period: 3
        fastd_matype: 0
    Outputs:
        fastk
        fastd
    """
    return _STOCHRSI(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod,
        fastk_period,
        fastd_period,
        fastd_matype
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _STOCHRSI( np.ndarray real, int timeperiod=14, int fastk_period=5, int fastd_period=3, int fastd_matype=0 ):
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

def SUB( real0 not None, real1 not None ):
    """ SUB(real0, real1)

    Vector Arithmetic Substraction (Math Operators)

    Inputs:
        real0: (np.ndarray or pd.Series)
        real1: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _SUB(
        real0.values if isinstance(real0, __PANDAS_SERIES) else real0,
        real1.values if isinstance(real1, __PANDAS_SERIES) else real1
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _SUB( np.ndarray real0, np.ndarray real1 ):
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

def SUM( real not None, timeperiod=30 ):
    """ SUM(real[, timeperiod=?])

    Summation (Math Operators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _SUM(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _SUM( np.ndarray real, int timeperiod=30 ):
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

def T3( real not None, timeperiod=5, vfactor=0.7 ):
    """ T3(real[, timeperiod=?, vfactor=?])

    Triple Exponential Moving Average (T3) (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 5
        vfactor: 0.7
    Outputs:
        real
    """
    return _T3(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod,
        vfactor
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _T3( np.ndarray real, int timeperiod=5, double vfactor=0.7 ):
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

def TAN( real not None ):
    """ TAN(real)

    Vector Trigonometric Tan (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _TAN(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _TAN( np.ndarray real ):
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

def TANH( real not None ):
    """ TANH(real)

    Vector Trigonometric Tanh (Math Transform)

    Inputs:
        real: (np.ndarray or pd.Series)
    Outputs:
        real
    """
    return _TANH(
        real.values if isinstance(real, __PANDAS_SERIES) else real
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _TANH( np.ndarray real ):
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

def TEMA( real not None, timeperiod=30 ):
    """ TEMA(real[, timeperiod=?])

    Triple Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _TEMA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _TEMA( np.ndarray real, int timeperiod=30 ):
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

def TRANGE( high not None, low not None, close not None ):
    """ TRANGE(high, low, close)

    True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    return _TRANGE(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _TRANGE( np.ndarray high, np.ndarray low, np.ndarray close ):
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

def TRIMA( real not None, timeperiod=30 ):
    """ TRIMA(real[, timeperiod=?])

    Triangular Moving Average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _TRIMA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _TRIMA( np.ndarray real, int timeperiod=30 ):
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

def TRIX( real not None, timeperiod=30 ):
    """ TRIX(real[, timeperiod=?])

    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (Momentum Indicators)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _TRIX(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _TRIX( np.ndarray real, int timeperiod=30 ):
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

def TSF( real not None, timeperiod=14 ):
    """ TSF(real[, timeperiod=?])

    Time Series Forecast (Statistic Functions)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _TSF(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _TSF( np.ndarray real, int timeperiod=14 ):
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

def TYPPRICE( high not None, low not None, close not None ):
    """ TYPPRICE(high, low, close)

    Typical Price (Price Transform)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    return _TYPPRICE(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _TYPPRICE( np.ndarray high, np.ndarray low, np.ndarray close ):
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

def ULTOSC( high not None, low not None, close not None, timeperiod1=7, timeperiod2=14, timeperiod3=28 ):
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
    return _ULTOSC(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod1,
        timeperiod2,
        timeperiod3
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _ULTOSC( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod1=7, int timeperiod2=14, int timeperiod3=28 ):
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

def VAR( real not None, timeperiod=5, nbdev=1.0 ):
    """ VAR(real[, timeperiod=?, nbdev=?])

    Variance (Statistic Functions)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 5
        nbdev: 1
    Outputs:
        real
    """
    return _VAR(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod,
        nbdev
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _VAR( np.ndarray real, int timeperiod=5, double nbdev=1 ):
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

def WCLPRICE( high not None, low not None, close not None ):
    """ WCLPRICE(high, low, close)

    Weighted Close Price (Price Transform)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    return _WCLPRICE(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _WCLPRICE( np.ndarray high, np.ndarray low, np.ndarray close ):
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

def WILLR( high not None, low not None, close not None, timeperiod=14 ):
    """ WILLR(high, low, close[, timeperiod=?])

    Williams' %R (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    return _WILLR(
        high.values if isinstance(high, __PANDAS_SERIES) else high,
        low.values if isinstance(low, __PANDAS_SERIES) else low,
        close.values if isinstance(close, __PANDAS_SERIES) else close,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _WILLR( np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod=14 ):
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

def WMA( real not None, timeperiod=30 ):
    """ WMA(real[, timeperiod=?])

    Weighted Moving Average (Overlap Studies)

    Inputs:
        real: (np.ndarray or pd.Series)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    return _WMA(
        real.values if isinstance(real, __PANDAS_SERIES) else real,
        timeperiod
    )

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef _WMA( np.ndarray real, int timeperiod=30 ):
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
