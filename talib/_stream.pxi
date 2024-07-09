cimport numpy as np
from cython import boundscheck, wraparound
cimport _ta_lib as lib
from _ta_lib cimport TA_RetCode
# NOTE: _ta_check_success, NaN are defined in common.pxi

np.import_array() # Initialize the NumPy C API

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ACOS( np.ndarray real not None ):
    """ ACOS(real)

    Vector Trigonometric ACos (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ACOS( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ACOS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AD( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None ):
    """ AD(high, low, close, volume)

    Chaikin A/D Line (Volume Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    volume = check_array(volume)
    volume_data = <double*>volume.data
    length = check_length4(high, low, close, volume)
    outreal = NaN
    retCode = lib.TA_AD( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , volume_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ADD( np.ndarray real0 not None , np.ndarray real1 not None ):
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
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_ADD( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ADOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int fastperiod=-2**31 , int slowperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    volume = check_array(volume)
    volume_data = <double*>volume.data
    length = check_length4(high, low, close, volume)
    outreal = NaN
    retCode = lib.TA_ADOSC( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , volume_data , fastperiod , slowperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ADX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_ADX( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ADXR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_ADXR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ADXR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_APO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_APO( <int>(length) - 1 , <int>(length) - 1 , real_data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_APO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AROON( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outaroondown
        double outaroonup
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outaroondown = NaN
    outaroonup = NaN
    retCode = lib.TA_AROON( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outaroondown , &outaroonup )
    _ta_check_success("TA_AROON", retCode)
    return outaroondown , outaroonup 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AROONOSC( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_AROONOSC( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AROONOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ASIN( np.ndarray real not None ):
    """ ASIN(real)

    Vector Trigonometric ASin (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ASIN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ASIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ATAN( np.ndarray real not None ):
    """ ATAN(real)

    Vector Trigonometric ATan (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ATAN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ATAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_ATR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_AVGPRICE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ AVGPRICE(open, high, low, close)

    Average Price (Price Transform)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outreal = NaN
    retCode = lib.TA_AVGPRICE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_AVGPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_BBANDS( np.ndarray real not None , int timeperiod=-2**31 , double nbdevup=-4e37 , double nbdevdn=-4e37 , int matype=0 ):
    """ BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])

    Bollinger Bands (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdevup: 2.0
        nbdevdn: 2.0
        matype: 0 (Simple Moving Average)
    Outputs:
        upperband
        middleband
        lowerband
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outrealupperband
        double outrealmiddleband
        double outreallowerband
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outrealupperband = NaN
    outrealmiddleband = NaN
    outreallowerband = NaN
    retCode = lib.TA_BBANDS( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , nbdevup , nbdevdn , matype , &outbegidx , &outnbelement , &outrealupperband , &outrealmiddleband , &outreallowerband )
    _ta_check_success("TA_BBANDS", retCode)
    return outrealupperband , outrealmiddleband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_BETA( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_BETA( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_BETA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_BOP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ BOP(open, high, low, close)

    Balance Of Power (Momentum Indicators)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outreal = NaN
    retCode = lib.TA_BOP( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_BOP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CCI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_CCI( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CCI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDL2CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL2CROWS(open, high, low, close)

    Two Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDL2CROWS( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL2CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDL3BLACKCROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3BLACKCROWS(open, high, low, close)

    Three Black Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDL3BLACKCROWS( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3BLACKCROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDL3INSIDE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3INSIDE(open, high, low, close)

    Three Inside Up/Down (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDL3INSIDE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3INSIDE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDL3LINESTRIKE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3LINESTRIKE(open, high, low, close)

    Three-Line Strike  (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDL3LINESTRIKE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3LINESTRIKE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDL3OUTSIDE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3OUTSIDE(open, high, low, close)

    Three Outside Up/Down (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDL3OUTSIDE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3OUTSIDE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDL3STARSINSOUTH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3STARSINSOUTH(open, high, low, close)

    Three Stars In The South (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDL3STARSINSOUTH( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3STARSINSOUTH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDL3WHITESOLDIERS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDL3WHITESOLDIERS(open, high, low, close)

    Three Advancing White Soldiers (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDL3WHITESOLDIERS( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDL3WHITESOLDIERS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLABANDONEDBABY( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
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
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLABANDONEDBABY( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLABANDONEDBABY", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLADVANCEBLOCK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLADVANCEBLOCK(open, high, low, close)

    Advance Block (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLADVANCEBLOCK( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLADVANCEBLOCK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLBELTHOLD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLBELTHOLD(open, high, low, close)

    Belt-hold (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLBELTHOLD( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLBELTHOLD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLBREAKAWAY( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLBREAKAWAY(open, high, low, close)

    Breakaway (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLBREAKAWAY( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLBREAKAWAY", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLCLOSINGMARUBOZU( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLCLOSINGMARUBOZU(open, high, low, close)

    Closing Marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLCLOSINGMARUBOZU( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLCLOSINGMARUBOZU", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLCONCEALBABYSWALL( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLCONCEALBABYSWALL(open, high, low, close)

    Concealing Baby Swallow (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLCONCEALBABYSWALL( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLCONCEALBABYSWALL", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLCOUNTERATTACK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLCOUNTERATTACK(open, high, low, close)

    Counterattack (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLCOUNTERATTACK( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLCOUNTERATTACK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLDARKCLOUDCOVER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.5 ):
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
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLDARKCLOUDCOVER( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLDARKCLOUDCOVER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLDOJI(open, high, low, close)

    Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLDOJI( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLDOJISTAR(open, high, low, close)

    Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLDOJISTAR( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLDRAGONFLYDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLDRAGONFLYDOJI(open, high, low, close)

    Dragonfly Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLDRAGONFLYDOJI( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLDRAGONFLYDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLENGULFING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLENGULFING(open, high, low, close)

    Engulfing Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLENGULFING( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLENGULFING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLEVENINGDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
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
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLEVENINGDOJISTAR( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLEVENINGDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLEVENINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
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
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLEVENINGSTAR( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLEVENINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLGAPSIDESIDEWHITE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLGAPSIDESIDEWHITE(open, high, low, close)

    Up/Down-gap side-by-side white lines (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLGAPSIDESIDEWHITE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLGAPSIDESIDEWHITE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLGRAVESTONEDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLGRAVESTONEDOJI(open, high, low, close)

    Gravestone Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLGRAVESTONEDOJI( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLGRAVESTONEDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLHAMMER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHAMMER(open, high, low, close)

    Hammer (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLHAMMER( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHAMMER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLHANGINGMAN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHANGINGMAN(open, high, low, close)

    Hanging Man (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLHANGINGMAN( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHANGINGMAN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLHARAMI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHARAMI(open, high, low, close)

    Harami Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLHARAMI( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHARAMI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLHARAMICROSS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHARAMICROSS(open, high, low, close)

    Harami Cross Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLHARAMICROSS( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHARAMICROSS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLHIGHWAVE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHIGHWAVE(open, high, low, close)

    High-Wave Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLHIGHWAVE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHIGHWAVE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLHIKKAKE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHIKKAKE(open, high, low, close)

    Hikkake Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLHIKKAKE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHIKKAKE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLHIKKAKEMOD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHIKKAKEMOD(open, high, low, close)

    Modified Hikkake Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLHIKKAKEMOD( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHIKKAKEMOD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLHOMINGPIGEON( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLHOMINGPIGEON(open, high, low, close)

    Homing Pigeon (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLHOMINGPIGEON( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLHOMINGPIGEON", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLIDENTICAL3CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLIDENTICAL3CROWS(open, high, low, close)

    Identical Three Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLIDENTICAL3CROWS( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLIDENTICAL3CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLINNECK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLINNECK(open, high, low, close)

    In-Neck Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLINNECK( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLINNECK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLINVERTEDHAMMER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLINVERTEDHAMMER(open, high, low, close)

    Inverted Hammer (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLINVERTEDHAMMER( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLINVERTEDHAMMER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLKICKING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLKICKING(open, high, low, close)

    Kicking (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLKICKING( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLKICKING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLKICKINGBYLENGTH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLKICKINGBYLENGTH(open, high, low, close)

    Kicking - bull/bear determined by the longer marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLKICKINGBYLENGTH( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLKICKINGBYLENGTH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLLADDERBOTTOM( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLLADDERBOTTOM(open, high, low, close)

    Ladder Bottom (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLLADDERBOTTOM( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLLADDERBOTTOM", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLLONGLEGGEDDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLLONGLEGGEDDOJI(open, high, low, close)

    Long Legged Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLLONGLEGGEDDOJI( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLLONGLEGGEDDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLLONGLINE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLLONGLINE(open, high, low, close)

    Long Line Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLLONGLINE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLLONGLINE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLMARUBOZU( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLMARUBOZU(open, high, low, close)

    Marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLMARUBOZU( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMARUBOZU", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLMATCHINGLOW( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLMATCHINGLOW(open, high, low, close)

    Matching Low (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLMATCHINGLOW( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMATCHINGLOW", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLMATHOLD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.5 ):
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
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLMATHOLD( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMATHOLD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLMORNINGDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
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
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLMORNINGDOJISTAR( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMORNINGDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLMORNINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=0.3 ):
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
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLMORNINGSTAR( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , penetration , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLMORNINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLONNECK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLONNECK(open, high, low, close)

    On-Neck Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLONNECK( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLONNECK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLPIERCING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLPIERCING(open, high, low, close)

    Piercing Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLPIERCING( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLPIERCING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLRICKSHAWMAN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLRICKSHAWMAN(open, high, low, close)

    Rickshaw Man (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLRICKSHAWMAN( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLRICKSHAWMAN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLRISEFALL3METHODS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLRISEFALL3METHODS(open, high, low, close)

    Rising/Falling Three Methods (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLRISEFALL3METHODS( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLRISEFALL3METHODS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLSEPARATINGLINES( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSEPARATINGLINES(open, high, low, close)

    Separating Lines (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLSEPARATINGLINES( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSEPARATINGLINES", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLSHOOTINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSHOOTINGSTAR(open, high, low, close)

    Shooting Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLSHOOTINGSTAR( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSHOOTINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLSHORTLINE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSHORTLINE(open, high, low, close)

    Short Line Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLSHORTLINE( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSHORTLINE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLSPINNINGTOP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSPINNINGTOP(open, high, low, close)

    Spinning Top (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLSPINNINGTOP( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSPINNINGTOP", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLSTALLEDPATTERN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSTALLEDPATTERN(open, high, low, close)

    Stalled Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLSTALLEDPATTERN( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSTALLEDPATTERN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLSTICKSANDWICH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLSTICKSANDWICH(open, high, low, close)

    Stick Sandwich (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLSTICKSANDWICH( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLSTICKSANDWICH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLTAKURI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLTAKURI(open, high, low, close)

    Takuri (Dragonfly Doji with very long lower shadow) (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLTAKURI( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLTAKURI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLTASUKIGAP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLTASUKIGAP(open, high, low, close)

    Tasuki Gap (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLTASUKIGAP( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLTASUKIGAP", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLTHRUSTING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLTHRUSTING(open, high, low, close)

    Thrusting Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLTHRUSTING( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLTHRUSTING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLTRISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLTRISTAR(open, high, low, close)

    Tristar Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLTRISTAR( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLTRISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLUNIQUE3RIVER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLUNIQUE3RIVER(open, high, low, close)

    Unique 3 River (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLUNIQUE3RIVER( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLUNIQUE3RIVER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLUPSIDEGAP2CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLUPSIDEGAP2CROWS(open, high, low, close)

    Upside Gap Two Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLUPSIDEGAP2CROWS( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLUPSIDEGAP2CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CDLXSIDEGAP3METHODS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ CDLXSIDEGAP3METHODS(open, high, low, close)

    Upside/Downside Gap Three Methods (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        int outinteger
    open = check_array(open)
    open_data = <double*>open.data
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length4(open, high, low, close)
    outinteger = 0
    retCode = lib.TA_CDLXSIDEGAP3METHODS( <int>(length) - 1 , <int>(length) - 1 , open_data , high_data , low_data , close_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_CDLXSIDEGAP3METHODS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CEIL( np.ndarray real not None ):
    """ CEIL(real)

    Vector Ceil (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_CEIL( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CEIL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CMO( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_CMO( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CMO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_CORREL( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_CORREL( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_CORREL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_COS( np.ndarray real not None ):
    """ COS(real)

    Vector Trigonometric Cos (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_COS( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_COS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_COSH( np.ndarray real not None ):
    """ COSH(real)

    Vector Trigonometric Cosh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_COSH( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_COSH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_DEMA( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_DEMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_DIV( np.ndarray real0 not None , np.ndarray real1 not None ):
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
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_DIV( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DIV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_DX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_DX( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_DX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_EMA( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_EMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_EMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_EXP( np.ndarray real not None ):
    """ EXP(real)

    Vector Arithmetic Exp (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_EXP( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_EXP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_FLOOR( np.ndarray real not None ):
    """ FLOOR(real)

    Vector Floor (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_FLOOR( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_FLOOR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_DCPERIOD( np.ndarray real not None ):
    """ HT_DCPERIOD(real)

    Hilbert Transform - Dominant Cycle Period (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_DCPERIOD( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_DCPERIOD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_DCPHASE( np.ndarray real not None ):
    """ HT_DCPHASE(real)

    Hilbert Transform - Dominant Cycle Phase (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_DCPHASE( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_DCPHASE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_PHASOR( np.ndarray real not None ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outinphase
        double outquadrature
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinphase = NaN
    outquadrature = NaN
    retCode = lib.TA_HT_PHASOR( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outinphase , &outquadrature )
    _ta_check_success("TA_HT_PHASOR", retCode)
    return outinphase , outquadrature 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_SINE( np.ndarray real not None ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outsine
        double outleadsine
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outsine = NaN
    outleadsine = NaN
    retCode = lib.TA_HT_SINE( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outsine , &outleadsine )
    _ta_check_success("TA_HT_SINE", retCode)
    return outsine , outleadsine 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_TRENDLINE( np.ndarray real not None ):
    """ HT_TRENDLINE(real)

    Hilbert Transform - Instantaneous Trendline (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_HT_TRENDLINE( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_HT_TRENDLINE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_HT_TRENDMODE( np.ndarray real not None ):
    """ HT_TRENDMODE(real)

    Hilbert Transform - Trend vs Cycle Mode (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_HT_TRENDMODE( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_HT_TRENDMODE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_KAMA( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_KAMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_KAMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LINEARREG( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LINEARREG_ANGLE( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_ANGLE( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_ANGLE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LINEARREG_INTERCEPT( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_INTERCEPT( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_INTERCEPT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LINEARREG_SLOPE( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LINEARREG_SLOPE( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LINEARREG_SLOPE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LN( np.ndarray real not None ):
    """ LN(real)

    Vector Log Natural (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_LOG10( np.ndarray real not None ):
    """ LOG10(real)

    Vector Log10 (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_LOG10( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_LOG10", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MA( np.ndarray real not None , int timeperiod=-2**31 , int matype=0 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MACD( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int signalperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACD( <int>(length) - 1 , <int>(length) - 1 , real_data , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACD", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MACDEXT( np.ndarray real not None , int fastperiod=-2**31 , int fastmatype=0 , int slowperiod=-2**31 , int slowmatype=0 , int signalperiod=-2**31 , int signalmatype=0 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACDEXT( <int>(length) - 1 , <int>(length) - 1 , real_data , fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACDEXT", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MACDFIX( np.ndarray real not None , int signalperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmacd
        double outmacdsignal
        double outmacdhist
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmacd = NaN
    outmacdsignal = NaN
    outmacdhist = NaN
    retCode = lib.TA_MACDFIX( <int>(length) - 1 , <int>(length) - 1 , real_data , signalperiod , &outbegidx , &outnbelement , &outmacd , &outmacdsignal , &outmacdhist )
    _ta_check_success("TA_MACDFIX", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MAMA( np.ndarray real not None , double fastlimit=-4e37 , double slowlimit=-4e37 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmama
        double outfama
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmama = NaN
    outfama = NaN
    retCode = lib.TA_MAMA( <int>(length) - 1 , <int>(length) - 1 , real_data , fastlimit , slowlimit , &outbegidx , &outnbelement , &outmama , &outfama )
    _ta_check_success("TA_MAMA", retCode)
    return outmama , outfama 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MAVP( np.ndarray real not None , np.ndarray periods not None , int minperiod=-2**31 , int maxperiod=-2**31 , int matype=0 ):
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
        TA_RetCode retCode
        double* real_data
        double* periods_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    periods = check_array(periods)
    periods_data = <double*>periods.data
    length = check_length2(real, periods)
    outreal = NaN
    retCode = lib.TA_MAVP( <int>(length) - 1 , <int>(length) - 1 , real_data , periods_data , minperiod , maxperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MAVP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MAX( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MAX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MAX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_MAXINDEX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_MAXINDEX", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MEDPRICE( np.ndarray high not None , np.ndarray low not None ):
    """ MEDPRICE(high, low)

    Median Price (Price Transform)

    Inputs:
        prices: ['high', 'low']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_MEDPRICE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MEDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MFI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    volume = check_array(volume)
    volume_data = <double*>volume.data
    length = check_length4(high, low, close, volume)
    outreal = NaN
    retCode = lib.TA_MFI( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , volume_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MFI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MIDPOINT( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MIDPOINT( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIDPOINT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MIDPRICE( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_MIDPRICE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MIN( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MIN( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MININDEX( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outinteger
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outinteger = 0
    retCode = lib.TA_MININDEX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outinteger )
    _ta_check_success("TA_MININDEX", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MINMAX( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outmin
        double outmax
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outmin = NaN
    outmax = NaN
    retCode = lib.TA_MINMAX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outmin , &outmax )
    _ta_check_success("TA_MINMAX", retCode)
    return outmin , outmax 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MINMAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        int outminidx
        int outmaxidx
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outminidx = 0
    outmaxidx = 0
    retCode = lib.TA_MINMAXINDEX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outminidx , &outmaxidx )
    _ta_check_success("TA_MINMAXINDEX", retCode)
    return outminidx , outmaxidx 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MINUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_MINUS_DI( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MINUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MINUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_MINUS_DM( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MINUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MOM( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_MOM( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MOM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_MULT( np.ndarray real0 not None , np.ndarray real1 not None ):
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
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_MULT( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_MULT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_NATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_NATR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_NATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_OBV( np.ndarray real not None , np.ndarray volume not None ):
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
        TA_RetCode retCode
        double* real_data
        double* volume_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    volume = check_array(volume)
    volume_data = <double*>volume.data
    length = check_length2(real, volume)
    outreal = NaN
    retCode = lib.TA_OBV( <int>(length) - 1 , <int>(length) - 1 , real_data , volume_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_OBV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_PLUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_PLUS_DI( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PLUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_PLUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_PLUS_DM( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PLUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_PPO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_PPO( <int>(length) - 1 , <int>(length) - 1 , real_data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_PPO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ROC( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROC( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ROCP( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCP( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ROCR( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCR( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ROCR100( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_ROCR100( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ROCR100", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_RSI( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_RSI( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_RSI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SAR( np.ndarray high not None , np.ndarray low not None , double acceleration=0.02 , double maximum=0.2 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_SAR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , acceleration , maximum , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SAREXT( np.ndarray high not None , np.ndarray low not None , double startvalue=-4e37 , double offsetonreverse=-4e37 , double accelerationinitlong=-4e37 , double accelerationlong=-4e37 , double accelerationmaxlong=-4e37 , double accelerationinitshort=-4e37 , double accelerationshort=-4e37 , double accelerationmaxshort=-4e37 ):
    """ SAREXT(high, low[, startvalue=?, offsetonreverse=?, accelerationinitlong=?, accelerationlong=?, accelerationmaxlong=?, accelerationinitshort=?, accelerationshort=?, accelerationmaxshort=?])

    Parabolic SAR - Extended (Overlap Studies)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        startvalue: 0.0
        offsetonreverse: 0.0
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    length = check_length2(high, low)
    outreal = NaN
    retCode = lib.TA_SAREXT( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SAREXT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SIN( np.ndarray real not None ):
    """ SIN(real)

    Vector Trigonometric Sin (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SIN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SINH( np.ndarray real not None ):
    """ SINH(real)

    Vector Trigonometric Sinh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SINH( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SINH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SMA( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SQRT( np.ndarray real not None ):
    """ SQRT(real)

    Vector Square Root (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SQRT( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SQRT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_STDDEV( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """ STDDEV(real[, timeperiod=?, nbdev=?])

    Standard Deviation (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdev: 1.0
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_STDDEV( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , nbdev , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_STDDEV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_STOCH( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int slowk_period=-2**31 , int slowk_matype=0 , int slowd_period=-2**31 , int slowd_matype=0 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outslowk
        double outslowd
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outslowk = NaN
    outslowd = NaN
    retCode = lib.TA_STOCH( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , &outslowk , &outslowd )
    _ta_check_success("TA_STOCH", retCode)
    return outslowk , outslowd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_STOCHF( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outfastk
        double outfastd
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outfastk = NaN
    outfastd = NaN
    retCode = lib.TA_STOCHF( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , &outfastk , &outfastd )
    _ta_check_success("TA_STOCHF", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_STOCHRSI( np.ndarray real not None , int timeperiod=-2**31 , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outfastk
        double outfastd
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outfastk = NaN
    outfastd = NaN
    retCode = lib.TA_STOCHRSI( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , &outfastk , &outfastd )
    _ta_check_success("TA_STOCHRSI", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SUB( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ SUB(real0, real1)

    Vector Arithmetic Subtraction (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        double outreal
    real0 = check_array(real0)
    real0_data = <double*>real0.data
    real1 = check_array(real1)
    real1_data = <double*>real1.data
    length = check_length2(real0, real1)
    outreal = NaN
    retCode = lib.TA_SUB( <int>(length) - 1 , <int>(length) - 1 , real0_data , real1_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SUB", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_SUM( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_SUM( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_SUM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_T3( np.ndarray real not None , int timeperiod=-2**31 , double vfactor=-4e37 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_T3( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , vfactor , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_T3", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TAN( np.ndarray real not None ):
    """ TAN(real)

    Vector Trigonometric Tan (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TAN( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TANH( np.ndarray real not None ):
    """ TANH(real)

    Vector Trigonometric Tanh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TANH( <int>(length) - 1 , <int>(length) - 1 , real_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TANH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TEMA( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TEMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TRANGE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ TRANGE(high, low, close)

    True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_TRANGE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRANGE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TRIMA( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TRIMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRIMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TRIX( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TRIX( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TRIX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TSF( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_TSF( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TSF", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_TYPPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ TYPPRICE(high, low, close)

    Typical Price (Price Transform)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_TYPPRICE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_TYPPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_ULTOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod1=-2**31 , int timeperiod2=-2**31 , int timeperiod3=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_ULTOSC( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod1 , timeperiod2 , timeperiod3 , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_ULTOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_VAR( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """ VAR(real[, timeperiod=?, nbdev=?])

    Variance (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdev: 1.0
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_VAR( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , nbdev , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_VAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_WCLPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """ WCLPRICE(high, low, close)

    Weighted Close Price (Price Transform)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_WCLPRICE( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WCLPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_WILLR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        double outreal
    high = check_array(high)
    high_data = <double*>high.data
    low = check_array(low)
    low_data = <double*>low.data
    close = check_array(close)
    close_data = <double*>close.data
    length = check_length3(high, low, close)
    outreal = NaN
    retCode = lib.TA_WILLR( <int>(length) - 1 , <int>(length) - 1 , high_data , low_data , close_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WILLR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def stream_WMA( np.ndarray real not None , int timeperiod=-2**31 ):
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
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        double outreal
    real = check_array(real)
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = NaN
    retCode = lib.TA_WMA( <int>(length) - 1 , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , &outreal )
    _ta_check_success("TA_WMA", retCode)
    return outreal 

