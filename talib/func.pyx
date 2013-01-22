from numpy import nan
from cython import boundscheck, wraparound
cimport numpy as np

ctypedef np.double_t double_t
ctypedef np.int32_t int32_t

ctypedef int TA_RetCode
ctypedef int TA_MAType

# TA_RetCode enums
RetCodes = {
    0: 'Success',
    1: 'Library Not Initialized',
    2: 'Bad Parameter',
    3: 'Allocation Error',
    4: 'Group Not Found',
    5: 'Function Not Found',
    6: 'Invalid Handle',
    7: 'Invalid Parameter Holder',
    8: 'Invalid Parameter Holder Type',
    9: 'Invalid Parameter Function',
   10: 'Input Not All Initialized',
   11: 'Output Not All Initialized',
   12: 'Out-of-Range Start Index',
   13: 'Out-of-Range End Index',
   14: 'Invalid List Type',
   15: 'Bad Object',
   16: 'Not Supported',
 5000: 'Internal Error',
65535: 'Unknown Error',
}

cdef double NaN = nan

cdef extern from "math.h":
    bint isnan(double x)

cdef extern from "numpy/arrayobject.h":
    int PyArray_TYPE(np.ndarray)
    object PyArray_EMPTY(int, np.npy_intp*, int, int)
    int PyArray_FLAGS(np.ndarray)
    object PyArray_GETCONTIGUOUS(np.ndarray)

np.import_array() # Initialize the NumPy C API

# extract the needed part of ta_libc.h that I will use in the interface
cdef extern from "ta-lib/ta_libc.h":
    enum: TA_SUCCESS
    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()
    char *TA_GetVersionString()
    TA_RetCode TA_ACOS( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ACOS_Lookback(  )
    TA_RetCode TA_AD( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[],  double inVolume[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_AD_Lookback(  )
    TA_RetCode TA_ADD( int startIdx, int endIdx,  double inReal0[],  double inReal1[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ADD_Lookback(  )
    TA_RetCode TA_ADOSC( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[],  double inVolume[], int optInFastPeriod, int optInSlowPeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ADOSC_Lookback( int optInFastPeriod, int optInSlowPeriod )
    TA_RetCode TA_ADX( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ADX_Lookback( int optInTimePeriod )
    TA_RetCode TA_ADXR( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ADXR_Lookback( int optInTimePeriod )
    TA_RetCode TA_APO( int startIdx, int endIdx,  double inReal[], int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_APO_Lookback( int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType )
    TA_RetCode TA_AROON( int startIdx, int endIdx,  double inHigh[],  double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outAroonDown[], double outAroonUp[] )
    int TA_AROON_Lookback( int optInTimePeriod )
    TA_RetCode TA_AROONOSC( int startIdx, int endIdx,  double inHigh[],  double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_AROONOSC_Lookback( int optInTimePeriod )
    TA_RetCode TA_ASIN( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ASIN_Lookback(  )
    TA_RetCode TA_ATAN( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ATAN_Lookback(  )
    TA_RetCode TA_ATR( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ATR_Lookback( int optInTimePeriod )
    TA_RetCode TA_AVGPRICE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_AVGPRICE_Lookback(  )
    TA_RetCode TA_BBANDS( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, double optInNbDevUp, double optInNbDevDn, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outRealUpperBand[], double outRealMiddleBand[], double outRealLowerBand[] )
    int TA_BBANDS_Lookback( int optInTimePeriod, double optInNbDevUp, double optInNbDevDn, TA_MAType optInMAType )
    TA_RetCode TA_BETA( int startIdx, int endIdx,  double inReal0[],  double inReal1[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_BETA_Lookback( int optInTimePeriod )
    TA_RetCode TA_BOP( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_BOP_Lookback(  )
    TA_RetCode TA_CCI( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_CCI_Lookback( int optInTimePeriod )
    TA_RetCode TA_CDL2CROWS( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDL2CROWS_Lookback(  )
    TA_RetCode TA_CDL3BLACKCROWS( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDL3BLACKCROWS_Lookback(  )
    TA_RetCode TA_CDL3INSIDE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDL3INSIDE_Lookback(  )
    TA_RetCode TA_CDL3LINESTRIKE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDL3LINESTRIKE_Lookback(  )
    TA_RetCode TA_CDL3OUTSIDE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDL3OUTSIDE_Lookback(  )
    TA_RetCode TA_CDL3STARSINSOUTH( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDL3STARSINSOUTH_Lookback(  )
    TA_RetCode TA_CDL3WHITESOLDIERS( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDL3WHITESOLDIERS_Lookback(  )
    TA_RetCode TA_CDLABANDONEDBABY( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLABANDONEDBABY_Lookback( double optInPenetration )
    TA_RetCode TA_CDLADVANCEBLOCK( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLADVANCEBLOCK_Lookback(  )
    TA_RetCode TA_CDLBELTHOLD( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLBELTHOLD_Lookback(  )
    TA_RetCode TA_CDLBREAKAWAY( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLBREAKAWAY_Lookback(  )
    TA_RetCode TA_CDLCLOSINGMARUBOZU( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLCLOSINGMARUBOZU_Lookback(  )
    TA_RetCode TA_CDLCONCEALBABYSWALL( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLCONCEALBABYSWALL_Lookback(  )
    TA_RetCode TA_CDLCOUNTERATTACK( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLCOUNTERATTACK_Lookback(  )
    TA_RetCode TA_CDLDARKCLOUDCOVER( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLDARKCLOUDCOVER_Lookback( double optInPenetration )
    TA_RetCode TA_CDLDOJI( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLDOJI_Lookback(  )
    TA_RetCode TA_CDLDOJISTAR( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLDOJISTAR_Lookback(  )
    TA_RetCode TA_CDLDRAGONFLYDOJI( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLDRAGONFLYDOJI_Lookback(  )
    TA_RetCode TA_CDLENGULFING( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLENGULFING_Lookback(  )
    TA_RetCode TA_CDLEVENINGDOJISTAR( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLEVENINGDOJISTAR_Lookback( double optInPenetration )
    TA_RetCode TA_CDLEVENINGSTAR( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLEVENINGSTAR_Lookback( double optInPenetration )
    TA_RetCode TA_CDLGAPSIDESIDEWHITE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLGAPSIDESIDEWHITE_Lookback(  )
    TA_RetCode TA_CDLGRAVESTONEDOJI( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLGRAVESTONEDOJI_Lookback(  )
    TA_RetCode TA_CDLHAMMER( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLHAMMER_Lookback(  )
    TA_RetCode TA_CDLHANGINGMAN( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLHANGINGMAN_Lookback(  )
    TA_RetCode TA_CDLHARAMI( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLHARAMI_Lookback(  )
    TA_RetCode TA_CDLHARAMICROSS( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLHARAMICROSS_Lookback(  )
    TA_RetCode TA_CDLHIGHWAVE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLHIGHWAVE_Lookback(  )
    TA_RetCode TA_CDLHIKKAKE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLHIKKAKE_Lookback(  )
    TA_RetCode TA_CDLHIKKAKEMOD( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLHIKKAKEMOD_Lookback(  )
    TA_RetCode TA_CDLHOMINGPIGEON( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLHOMINGPIGEON_Lookback(  )
    TA_RetCode TA_CDLIDENTICAL3CROWS( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLIDENTICAL3CROWS_Lookback(  )
    TA_RetCode TA_CDLINNECK( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLINNECK_Lookback(  )
    TA_RetCode TA_CDLINVERTEDHAMMER( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLINVERTEDHAMMER_Lookback(  )
    TA_RetCode TA_CDLKICKING( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLKICKING_Lookback(  )
    TA_RetCode TA_CDLKICKINGBYLENGTH( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLKICKINGBYLENGTH_Lookback(  )
    TA_RetCode TA_CDLLADDERBOTTOM( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLLADDERBOTTOM_Lookback(  )
    TA_RetCode TA_CDLLONGLEGGEDDOJI( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLLONGLEGGEDDOJI_Lookback(  )
    TA_RetCode TA_CDLLONGLINE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLLONGLINE_Lookback(  )
    TA_RetCode TA_CDLMARUBOZU( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLMARUBOZU_Lookback(  )
    TA_RetCode TA_CDLMATCHINGLOW( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLMATCHINGLOW_Lookback(  )
    TA_RetCode TA_CDLMATHOLD( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLMATHOLD_Lookback( double optInPenetration )
    TA_RetCode TA_CDLMORNINGDOJISTAR( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLMORNINGDOJISTAR_Lookback( double optInPenetration )
    TA_RetCode TA_CDLMORNINGSTAR( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLMORNINGSTAR_Lookback( double optInPenetration )
    TA_RetCode TA_CDLONNECK( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLONNECK_Lookback(  )
    TA_RetCode TA_CDLPIERCING( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLPIERCING_Lookback(  )
    TA_RetCode TA_CDLRICKSHAWMAN( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLRICKSHAWMAN_Lookback(  )
    TA_RetCode TA_CDLRISEFALL3METHODS( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLRISEFALL3METHODS_Lookback(  )
    TA_RetCode TA_CDLSEPARATINGLINES( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLSEPARATINGLINES_Lookback(  )
    TA_RetCode TA_CDLSHOOTINGSTAR( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLSHOOTINGSTAR_Lookback(  )
    TA_RetCode TA_CDLSHORTLINE( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLSHORTLINE_Lookback(  )
    TA_RetCode TA_CDLSPINNINGTOP( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLSPINNINGTOP_Lookback(  )
    TA_RetCode TA_CDLSTALLEDPATTERN( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLSTALLEDPATTERN_Lookback(  )
    TA_RetCode TA_CDLSTICKSANDWICH( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLSTICKSANDWICH_Lookback(  )
    TA_RetCode TA_CDLTAKURI( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLTAKURI_Lookback(  )
    TA_RetCode TA_CDLTASUKIGAP( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLTASUKIGAP_Lookback(  )
    TA_RetCode TA_CDLTHRUSTING( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLTHRUSTING_Lookback(  )
    TA_RetCode TA_CDLTRISTAR( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLTRISTAR_Lookback(  )
    TA_RetCode TA_CDLUNIQUE3RIVER( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLUNIQUE3RIVER_Lookback(  )
    TA_RetCode TA_CDLUPSIDEGAP2CROWS( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLUPSIDEGAP2CROWS_Lookback(  )
    TA_RetCode TA_CDLXSIDEGAP3METHODS( int startIdx, int endIdx,  double inOpen[],  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_CDLXSIDEGAP3METHODS_Lookback(  )
    TA_RetCode TA_CEIL( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_CEIL_Lookback(  )
    TA_RetCode TA_CMO( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_CMO_Lookback( int optInTimePeriod )
    TA_RetCode TA_CORREL( int startIdx, int endIdx,  double inReal0[],  double inReal1[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_CORREL_Lookback( int optInTimePeriod )
    TA_RetCode TA_COS( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_COS_Lookback(  )
    TA_RetCode TA_COSH( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_COSH_Lookback(  )
    TA_RetCode TA_DEMA( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_DEMA_Lookback( int optInTimePeriod )
    TA_RetCode TA_DIV( int startIdx, int endIdx,  double inReal0[],  double inReal1[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_DIV_Lookback(  )
    TA_RetCode TA_DX( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_DX_Lookback( int optInTimePeriod )
    TA_RetCode TA_EMA( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_EMA_Lookback( int optInTimePeriod )
    TA_RetCode TA_EXP( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_EXP_Lookback(  )
    TA_RetCode TA_FLOOR( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_FLOOR_Lookback(  )
    TA_RetCode TA_HT_DCPERIOD( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_HT_DCPERIOD_Lookback(  )
    TA_RetCode TA_HT_DCPHASE( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_HT_DCPHASE_Lookback(  )
    TA_RetCode TA_HT_PHASOR( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outInPhase[], double outQuadrature[] )
    int TA_HT_PHASOR_Lookback(  )
    TA_RetCode TA_HT_SINE( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outSine[], double outLeadSine[] )
    int TA_HT_SINE_Lookback(  )
    TA_RetCode TA_HT_TRENDLINE( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_HT_TRENDLINE_Lookback(  )
    TA_RetCode TA_HT_TRENDMODE( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_HT_TRENDMODE_Lookback(  )
    TA_RetCode TA_KAMA( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_KAMA_Lookback( int optInTimePeriod )
    TA_RetCode TA_LINEARREG( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_LINEARREG_Lookback( int optInTimePeriod )
    TA_RetCode TA_LINEARREG_ANGLE( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_LINEARREG_ANGLE_Lookback( int optInTimePeriod )
    TA_RetCode TA_LINEARREG_INTERCEPT( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_LINEARREG_INTERCEPT_Lookback( int optInTimePeriod )
    TA_RetCode TA_LINEARREG_SLOPE( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_LINEARREG_SLOPE_Lookback( int optInTimePeriod )
    TA_RetCode TA_LN( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_LN_Lookback(  )
    TA_RetCode TA_LOG10( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_LOG10_Lookback(  )
    TA_RetCode TA_MA( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MA_Lookback( int optInTimePeriod, TA_MAType optInMAType )
    TA_RetCode TA_MACD( int startIdx, int endIdx,  double inReal[], int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod, int *outBegIdx, int *outNBElement, double outMACD[], double outMACDSignal[], double outMACDHist[] )
    int TA_MACD_Lookback( int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod )
    TA_RetCode TA_MACDEXT( int startIdx, int endIdx,  double inReal[], int optInFastPeriod, TA_MAType optInFastMAType, int optInSlowPeriod, TA_MAType optInSlowMAType, int optInSignalPeriod, TA_MAType optInSignalMAType, int *outBegIdx, int *outNBElement, double outMACD[], double outMACDSignal[], double outMACDHist[] )
    int TA_MACDEXT_Lookback( int optInFastPeriod, TA_MAType optInFastMAType, int optInSlowPeriod, TA_MAType optInSlowMAType, int optInSignalPeriod, TA_MAType optInSignalMAType )
    TA_RetCode TA_MACDFIX( int startIdx, int endIdx,  double inReal[], int optInSignalPeriod, int *outBegIdx, int *outNBElement, double outMACD[], double outMACDSignal[], double outMACDHist[] )
    int TA_MACDFIX_Lookback( int optInSignalPeriod )
    TA_RetCode TA_MAMA( int startIdx, int endIdx,  double inReal[], double optInFastLimit, double optInSlowLimit, int *outBegIdx, int *outNBElement, double outMAMA[], double outFAMA[] )
    int TA_MAMA_Lookback( double optInFastLimit, double optInSlowLimit )
    TA_RetCode TA_MAVP( int startIdx, int endIdx,  double inReal[],  double inPeriods[], int optInMinPeriod, int optInMaxPeriod, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MAVP_Lookback( int optInMinPeriod, int optInMaxPeriod, TA_MAType optInMAType )
    TA_RetCode TA_MAX( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MAX_Lookback( int optInTimePeriod )
    TA_RetCode TA_MAXINDEX( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_MAXINDEX_Lookback( int optInTimePeriod )
    TA_RetCode TA_MEDPRICE( int startIdx, int endIdx,  double inHigh[],  double inLow[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MEDPRICE_Lookback(  )
    TA_RetCode TA_MFI( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[],  double inVolume[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MFI_Lookback( int optInTimePeriod )
    TA_RetCode TA_MIDPOINT( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MIDPOINT_Lookback( int optInTimePeriod )
    TA_RetCode TA_MIDPRICE( int startIdx, int endIdx,  double inHigh[],  double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MIDPRICE_Lookback( int optInTimePeriod )
    TA_RetCode TA_MIN( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MIN_Lookback( int optInTimePeriod )
    TA_RetCode TA_MININDEX( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, int outInteger[] )
    int TA_MININDEX_Lookback( int optInTimePeriod )
    TA_RetCode TA_MINMAX( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outMin[], double outMax[] )
    int TA_MINMAX_Lookback( int optInTimePeriod )
    TA_RetCode TA_MINMAXINDEX( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, int outMinIdx[], int outMaxIdx[] )
    int TA_MINMAXINDEX_Lookback( int optInTimePeriod )
    TA_RetCode TA_MINUS_DI( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MINUS_DI_Lookback( int optInTimePeriod )
    TA_RetCode TA_MINUS_DM( int startIdx, int endIdx,  double inHigh[],  double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MINUS_DM_Lookback( int optInTimePeriod )
    TA_RetCode TA_MOM( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MOM_Lookback( int optInTimePeriod )
    TA_RetCode TA_MULT( int startIdx, int endIdx,  double inReal0[],  double inReal1[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_MULT_Lookback(  )
    TA_RetCode TA_NATR( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_NATR_Lookback( int optInTimePeriod )
    TA_RetCode TA_OBV( int startIdx, int endIdx,  double inReal[],  double inVolume[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_OBV_Lookback(  )
    TA_RetCode TA_PLUS_DI( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_PLUS_DI_Lookback( int optInTimePeriod )
    TA_RetCode TA_PLUS_DM( int startIdx, int endIdx,  double inHigh[],  double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_PLUS_DM_Lookback( int optInTimePeriod )
    TA_RetCode TA_PPO( int startIdx, int endIdx,  double inReal[], int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_PPO_Lookback( int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType )
    TA_RetCode TA_ROC( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ROC_Lookback( int optInTimePeriod )
    TA_RetCode TA_ROCP( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ROCP_Lookback( int optInTimePeriod )
    TA_RetCode TA_ROCR( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ROCR_Lookback( int optInTimePeriod )
    TA_RetCode TA_ROCR100( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ROCR100_Lookback( int optInTimePeriod )
    TA_RetCode TA_RSI( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_RSI_Lookback( int optInTimePeriod )
    TA_RetCode TA_SAR( int startIdx, int endIdx,  double inHigh[],  double inLow[], double optInAcceleration, double optInMaximum, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_SAR_Lookback( double optInAcceleration, double optInMaximum )
    TA_RetCode TA_SAREXT( int startIdx, int endIdx,  double inHigh[],  double inLow[], double optInStartValue, double optInOffsetOnReverse, double optInAccelerationInitLong, double optInAccelerationLong, double optInAccelerationMaxLong, double optInAccelerationInitShort, double optInAccelerationShort, double optInAccelerationMaxShort, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_SAREXT_Lookback( double optInStartValue, double optInOffsetOnReverse, double optInAccelerationInitLong, double optInAccelerationLong, double optInAccelerationMaxLong, double optInAccelerationInitShort, double optInAccelerationShort, double optInAccelerationMaxShort )
    TA_RetCode TA_SIN( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_SIN_Lookback(  )
    TA_RetCode TA_SINH( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_SINH_Lookback(  )
    TA_RetCode TA_SMA( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_SMA_Lookback( int optInTimePeriod )
    TA_RetCode TA_SQRT( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_SQRT_Lookback(  )
    TA_RetCode TA_STDDEV( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, double optInNbDev, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_STDDEV_Lookback( int optInTimePeriod, double optInNbDev )
    TA_RetCode TA_STOCH( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInFastK_Period, int optInSlowK_Period, TA_MAType optInSlowK_MAType, int optInSlowD_Period, TA_MAType optInSlowD_MAType, int *outBegIdx, int *outNBElement, double outSlowK[], double outSlowD[] )
    int TA_STOCH_Lookback( int optInFastK_Period, int optInSlowK_Period, TA_MAType optInSlowK_MAType, int optInSlowD_Period, TA_MAType optInSlowD_MAType )
    TA_RetCode TA_STOCHF( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType, int *outBegIdx, int *outNBElement, double outFastK[], double outFastD[] )
    int TA_STOCHF_Lookback( int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType )
    TA_RetCode TA_STOCHRSI( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType, int *outBegIdx, int *outNBElement, double outFastK[], double outFastD[] )
    int TA_STOCHRSI_Lookback( int optInTimePeriod, int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType )
    TA_RetCode TA_SUB( int startIdx, int endIdx,  double inReal0[],  double inReal1[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_SUB_Lookback(  )
    TA_RetCode TA_SUM( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_SUM_Lookback( int optInTimePeriod )
    TA_RetCode TA_T3( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, double optInVFactor, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_T3_Lookback( int optInTimePeriod, double optInVFactor )
    TA_RetCode TA_TAN( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_TAN_Lookback(  )
    TA_RetCode TA_TANH( int startIdx, int endIdx,  double inReal[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_TANH_Lookback(  )
    TA_RetCode TA_TEMA( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_TEMA_Lookback( int optInTimePeriod )
    TA_RetCode TA_TRANGE( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_TRANGE_Lookback(  )
    TA_RetCode TA_TRIMA( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_TRIMA_Lookback( int optInTimePeriod )
    TA_RetCode TA_TRIX( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_TRIX_Lookback( int optInTimePeriod )
    TA_RetCode TA_TSF( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_TSF_Lookback( int optInTimePeriod )
    TA_RetCode TA_TYPPRICE( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_TYPPRICE_Lookback(  )
    TA_RetCode TA_ULTOSC( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod1, int optInTimePeriod2, int optInTimePeriod3, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_ULTOSC_Lookback( int optInTimePeriod1, int optInTimePeriod2, int optInTimePeriod3 )
    TA_RetCode TA_VAR( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, double optInNbDev, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_VAR_Lookback( int optInTimePeriod, double optInNbDev )
    TA_RetCode TA_WCLPRICE( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_WCLPRICE_Lookback(  )
    TA_RetCode TA_WILLR( int startIdx, int endIdx,  double inHigh[],  double inLow[],  double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_WILLR_Lookback( int optInTimePeriod )
    TA_RetCode TA_WMA( int startIdx, int endIdx,  double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[] )
    int TA_WMA_Lookback( int optInTimePeriod )


__version__ = TA_GetVersionString()

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ACOS( np.ndarray real not None ):
    """ACOS(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ACOS_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ACOS( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AD( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None ):
    """AD(high, low, close, volume)

    Chaikin A/D Line"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    assert PyArray_TYPE(volume) == np.NPY_DOUBLE, "volume is not double"
    assert volume.ndim == 1, "volume has wrong dimensions"
    if not (PyArray_FLAGS(volume) & np.NPY_C_CONTIGUOUS):
        volume = PyArray_GETCONTIGUOUS(volume)
    volume_data = <double*>volume.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_AD_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_AD( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , <double *>(volume_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADD( np.ndarray real0 not None , np.ndarray real1 not None ):
    """ADD(real0, real1)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real0) == np.NPY_DOUBLE, "real0 is not double"
    assert real0.ndim == 1, "real0 has wrong dimensions"
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    assert PyArray_TYPE(real1) == np.NPY_DOUBLE, "real1 is not double"
    assert real1.ndim == 1, "real1 has wrong dimensions"
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real0_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ADD_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ADD( 0 , endidx , <double *>(real0_data+begidx) , <double *>(real1_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int fastperiod=-2**31 , int slowperiod=-2**31 ):
    """ADOSC(high, low, close, volume[, fastperiod=?, slowperiod=?])

    Chaikin A/D Oscillator"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    assert PyArray_TYPE(volume) == np.NPY_DOUBLE, "volume is not double"
    assert volume.ndim == 1, "volume has wrong dimensions"
    if not (PyArray_FLAGS(volume) & np.NPY_C_CONTIGUOUS):
        volume = PyArray_GETCONTIGUOUS(volume)
    volume_data = <double*>volume.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ADOSC_Lookback( fastperiod , slowperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ADOSC( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , <double *>(volume_data+begidx) , fastperiod , slowperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ADX(high, low, close[, timeperiod=?])

    Average Directional Movement Index"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ADX_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ADX( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ADXR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ADXR(high, low, close[, timeperiod=?])

    Average Directional Movement Index Rating"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ADXR_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ADXR( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def APO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
    """APO(real[, fastperiod=?, slowperiod=?, matype=?])

    Absolute Price Oscillator"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_APO_Lookback( fastperiod , slowperiod , matype )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_APO( 0 , endidx , <double *>(real_data+begidx) , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AROON( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """AROON(high, low[, timeperiod=?])

    Aroon"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        np.ndarray outaroondown
        double* outaroondown_data
        np.ndarray outaroonup
        double* outaroonup_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_AROON_Lookback( timeperiod )
    outaroondown = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outaroondown_data = <double*>outaroondown.data
    for i from 0 <= i < min(lookback, length):
        outaroondown_data[i] = NaN
    outaroonup = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outaroonup_data = <double*>outaroonup.data
    for i from 0 <= i < min(lookback, length):
        outaroonup_data[i] = NaN
    retCode = TA_AROON( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outaroondown_data+lookback) , <double *>(outaroonup_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outaroondown , outaroonup

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AROONOSC( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """AROONOSC(high, low[, timeperiod=?])

    Aroon Oscillator"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_AROONOSC_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_AROONOSC( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ASIN( np.ndarray real not None ):
    """ASIN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ASIN_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ASIN( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ATAN( np.ndarray real not None ):
    """ATAN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ATAN_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ATAN( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """ATR(high, low, close[, timeperiod=?])

    Average True Range"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ATR_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ATR( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def AVGPRICE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """AVGPRICE(open, high, low, close)

    Average Price"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_AVGPRICE_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_AVGPRICE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BBANDS( np.ndarray real not None , int timeperiod=-2**31 , double nbdevup=-4e37 , double nbdevdn=-4e37 , int matype=0 ):
    """BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])

    Bollinger Bands"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outrealupperband
        double* outrealupperband_data
        np.ndarray outrealmiddleband
        double* outrealmiddleband_data
        np.ndarray outreallowerband
        double* outreallowerband_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_BBANDS_Lookback( timeperiod , nbdevup , nbdevdn , matype )
    outrealupperband = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outrealupperband_data = <double*>outrealupperband.data
    for i from 0 <= i < min(lookback, length):
        outrealupperband_data[i] = NaN
    outrealmiddleband = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outrealmiddleband_data = <double*>outrealmiddleband.data
    for i from 0 <= i < min(lookback, length):
        outrealmiddleband_data[i] = NaN
    outreallowerband = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreallowerband_data = <double*>outreallowerband.data
    for i from 0 <= i < min(lookback, length):
        outreallowerband_data[i] = NaN
    retCode = TA_BBANDS( 0 , endidx , <double *>(real_data+begidx) , timeperiod , nbdevup , nbdevdn , matype , &outbegidx , &outnbelement , <double *>(outrealupperband_data+lookback) , <double *>(outrealmiddleband_data+lookback) , <double *>(outreallowerband_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outrealupperband , outrealmiddleband , outreallowerband

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BETA( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
    """BETA(real0, real1[, timeperiod=?])

    Beta"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real0) == np.NPY_DOUBLE, "real0 is not double"
    assert real0.ndim == 1, "real0 has wrong dimensions"
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    assert PyArray_TYPE(real1) == np.NPY_DOUBLE, "real1 is not double"
    assert real1.ndim == 1, "real1 has wrong dimensions"
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real0_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_BETA_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_BETA( 0 , endidx , <double *>(real0_data+begidx) , <double *>(real1_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def BOP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """BOP(open, high, low, close)

    Balance Of Power"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_BOP_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_BOP( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CCI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """CCI(high, low, close[, timeperiod=?])

    Commodity Channel Index"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CCI_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_CCI( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL2CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDL2CROWS(open, high, low, close)

    Two Crows"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDL2CROWS_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDL2CROWS( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3BLACKCROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDL3BLACKCROWS(open, high, low, close)

    Three Black Crows"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDL3BLACKCROWS_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDL3BLACKCROWS( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3INSIDE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDL3INSIDE(open, high, low, close)

    Three Inside Up/Down"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDL3INSIDE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDL3INSIDE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3LINESTRIKE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDL3LINESTRIKE(open, high, low, close)

    Three-Line Strike """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDL3LINESTRIKE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDL3LINESTRIKE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3OUTSIDE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDL3OUTSIDE(open, high, low, close)

    Three Outside Up/Down"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDL3OUTSIDE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDL3OUTSIDE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3STARSINSOUTH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDL3STARSINSOUTH(open, high, low, close)

    Three Stars In The South"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDL3STARSINSOUTH_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDL3STARSINSOUTH( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDL3WHITESOLDIERS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDL3WHITESOLDIERS(open, high, low, close)

    Three Advancing White Soldiers"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDL3WHITESOLDIERS_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDL3WHITESOLDIERS( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLABANDONEDBABY( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=-4e37 ):
    """CDLABANDONEDBABY(open, high, low, close[, penetration=?])

    Abandoned Baby"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLABANDONEDBABY_Lookback( penetration )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLABANDONEDBABY( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , penetration , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLADVANCEBLOCK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLADVANCEBLOCK(open, high, low, close)

    Advance Block"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLADVANCEBLOCK_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLADVANCEBLOCK( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLBELTHOLD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLBELTHOLD(open, high, low, close)

    Belt-hold"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLBELTHOLD_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLBELTHOLD( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLBREAKAWAY( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLBREAKAWAY(open, high, low, close)

    Breakaway"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLBREAKAWAY_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLBREAKAWAY( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLCLOSINGMARUBOZU( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLCLOSINGMARUBOZU(open, high, low, close)

    Closing Marubozu"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLCLOSINGMARUBOZU_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLCLOSINGMARUBOZU( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLCONCEALBABYSWALL( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLCONCEALBABYSWALL(open, high, low, close)

    Concealing Baby Swallow"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLCONCEALBABYSWALL_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLCONCEALBABYSWALL( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLCOUNTERATTACK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLCOUNTERATTACK(open, high, low, close)

    Counterattack"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLCOUNTERATTACK_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLCOUNTERATTACK( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLDARKCLOUDCOVER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=-4e37 ):
    """CDLDARKCLOUDCOVER(open, high, low, close[, penetration=?])

    Dark Cloud Cover"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLDARKCLOUDCOVER_Lookback( penetration )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLDARKCLOUDCOVER( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , penetration , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLDOJI(open, high, low, close)

    Doji"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLDOJI_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLDOJI( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLDOJISTAR(open, high, low, close)

    Doji Star"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLDOJISTAR_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLDOJISTAR( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLDRAGONFLYDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLDRAGONFLYDOJI(open, high, low, close)

    Dragonfly Doji"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLDRAGONFLYDOJI_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLDRAGONFLYDOJI( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLENGULFING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLENGULFING(open, high, low, close)

    Engulfing Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLENGULFING_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLENGULFING( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLEVENINGDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=-4e37 ):
    """CDLEVENINGDOJISTAR(open, high, low, close[, penetration=?])

    Evening Doji Star"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLEVENINGDOJISTAR_Lookback( penetration )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLEVENINGDOJISTAR( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , penetration , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLEVENINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=-4e37 ):
    """CDLEVENINGSTAR(open, high, low, close[, penetration=?])

    Evening Star"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLEVENINGSTAR_Lookback( penetration )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLEVENINGSTAR( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , penetration , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLGAPSIDESIDEWHITE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLGAPSIDESIDEWHITE(open, high, low, close)

    Up/Down-gap side-by-side white lines"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLGAPSIDESIDEWHITE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLGAPSIDESIDEWHITE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLGRAVESTONEDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLGRAVESTONEDOJI(open, high, low, close)

    Gravestone Doji"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLGRAVESTONEDOJI_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLGRAVESTONEDOJI( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHAMMER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLHAMMER(open, high, low, close)

    Hammer"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLHAMMER_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLHAMMER( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHANGINGMAN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLHANGINGMAN(open, high, low, close)

    Hanging Man"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLHANGINGMAN_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLHANGINGMAN( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHARAMI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLHARAMI(open, high, low, close)

    Harami Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLHARAMI_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLHARAMI( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHARAMICROSS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLHARAMICROSS(open, high, low, close)

    Harami Cross Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLHARAMICROSS_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLHARAMICROSS( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHIGHWAVE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLHIGHWAVE(open, high, low, close)

    High-Wave Candle"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLHIGHWAVE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLHIGHWAVE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHIKKAKE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLHIKKAKE(open, high, low, close)

    Hikkake Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLHIKKAKE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLHIKKAKE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHIKKAKEMOD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLHIKKAKEMOD(open, high, low, close)

    Modified Hikkake Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLHIKKAKEMOD_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLHIKKAKEMOD( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLHOMINGPIGEON( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLHOMINGPIGEON(open, high, low, close)

    Homing Pigeon"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLHOMINGPIGEON_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLHOMINGPIGEON( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLIDENTICAL3CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLIDENTICAL3CROWS(open, high, low, close)

    Identical Three Crows"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLIDENTICAL3CROWS_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLIDENTICAL3CROWS( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLINNECK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLINNECK(open, high, low, close)

    In-Neck Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLINNECK_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLINNECK( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLINVERTEDHAMMER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLINVERTEDHAMMER(open, high, low, close)

    Inverted Hammer"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLINVERTEDHAMMER_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLINVERTEDHAMMER( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLKICKING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLKICKING(open, high, low, close)

    Kicking"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLKICKING_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLKICKING( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLKICKINGBYLENGTH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLKICKINGBYLENGTH(open, high, low, close)

    Kicking - bull/bear determined by the longer marubozu"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLKICKINGBYLENGTH_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLKICKINGBYLENGTH( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLLADDERBOTTOM( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLLADDERBOTTOM(open, high, low, close)

    Ladder Bottom"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLLADDERBOTTOM_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLLADDERBOTTOM( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLLONGLEGGEDDOJI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLLONGLEGGEDDOJI(open, high, low, close)

    Long Legged Doji"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLLONGLEGGEDDOJI_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLLONGLEGGEDDOJI( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLLONGLINE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLLONGLINE(open, high, low, close)

    Long Line Candle"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLLONGLINE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLLONGLINE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMARUBOZU( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLMARUBOZU(open, high, low, close)

    Marubozu"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLMARUBOZU_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLMARUBOZU( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMATCHINGLOW( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLMATCHINGLOW(open, high, low, close)

    Matching Low"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLMATCHINGLOW_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLMATCHINGLOW( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMATHOLD( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=-4e37 ):
    """CDLMATHOLD(open, high, low, close[, penetration=?])

    Mat Hold"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLMATHOLD_Lookback( penetration )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLMATHOLD( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , penetration , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMORNINGDOJISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=-4e37 ):
    """CDLMORNINGDOJISTAR(open, high, low, close[, penetration=?])

    Morning Doji Star"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLMORNINGDOJISTAR_Lookback( penetration )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLMORNINGDOJISTAR( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , penetration , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLMORNINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , double penetration=-4e37 ):
    """CDLMORNINGSTAR(open, high, low, close[, penetration=?])

    Morning Star"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLMORNINGSTAR_Lookback( penetration )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLMORNINGSTAR( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , penetration , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLONNECK( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLONNECK(open, high, low, close)

    On-Neck Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLONNECK_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLONNECK( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLPIERCING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLPIERCING(open, high, low, close)

    Piercing Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLPIERCING_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLPIERCING( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLRICKSHAWMAN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLRICKSHAWMAN(open, high, low, close)

    Rickshaw Man"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLRICKSHAWMAN_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLRICKSHAWMAN( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLRISEFALL3METHODS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLRISEFALL3METHODS(open, high, low, close)

    Rising/Falling Three Methods"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLRISEFALL3METHODS_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLRISEFALL3METHODS( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSEPARATINGLINES( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLSEPARATINGLINES(open, high, low, close)

    Separating Lines"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLSEPARATINGLINES_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLSEPARATINGLINES( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSHOOTINGSTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLSHOOTINGSTAR(open, high, low, close)

    Shooting Star"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLSHOOTINGSTAR_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLSHOOTINGSTAR( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSHORTLINE( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLSHORTLINE(open, high, low, close)

    Short Line Candle"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLSHORTLINE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLSHORTLINE( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSPINNINGTOP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLSPINNINGTOP(open, high, low, close)

    Spinning Top"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLSPINNINGTOP_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLSPINNINGTOP( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSTALLEDPATTERN( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLSTALLEDPATTERN(open, high, low, close)

    Stalled Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLSTALLEDPATTERN_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLSTALLEDPATTERN( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLSTICKSANDWICH( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLSTICKSANDWICH(open, high, low, close)

    Stick Sandwich"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLSTICKSANDWICH_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLSTICKSANDWICH( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLTAKURI( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLTAKURI(open, high, low, close)

    Takuri (Dragonfly Doji with very long lower shadow)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLTAKURI_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLTAKURI( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLTASUKIGAP( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLTASUKIGAP(open, high, low, close)

    Tasuki Gap"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLTASUKIGAP_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLTASUKIGAP( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLTHRUSTING( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLTHRUSTING(open, high, low, close)

    Thrusting Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLTHRUSTING_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLTHRUSTING( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLTRISTAR( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLTRISTAR(open, high, low, close)

    Tristar Pattern"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLTRISTAR_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLTRISTAR( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLUNIQUE3RIVER( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLUNIQUE3RIVER(open, high, low, close)

    Unique 3 River"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLUNIQUE3RIVER_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLUNIQUE3RIVER( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLUPSIDEGAP2CROWS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLUPSIDEGAP2CROWS(open, high, low, close)

    Upside Gap Two Crows"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLUPSIDEGAP2CROWS_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLUPSIDEGAP2CROWS( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CDLXSIDEGAP3METHODS( np.ndarray open not None , np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """CDLXSIDEGAP3METHODS(open, high, low, close)

    Upside/Downside Gap Three Methods"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* open_data
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(open) == np.NPY_DOUBLE, "open is not double"
    assert open.ndim == 1, "open has wrong dimensions"
    if not (PyArray_FLAGS(open) & np.NPY_C_CONTIGUOUS):
        open = PyArray_GETCONTIGUOUS(open)
    open_data = <double*>open.data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CDLXSIDEGAP3METHODS_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_CDLXSIDEGAP3METHODS( 0 , endidx , <double *>(open_data+begidx) , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CEIL( np.ndarray real not None ):
    """CEIL(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CEIL_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_CEIL( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CMO( np.ndarray real not None , int timeperiod=-2**31 ):
    """CMO(real[, timeperiod=?])

    Chande Momentum Oscillator"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CMO_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_CMO( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def CORREL( np.ndarray real0 not None , np.ndarray real1 not None , int timeperiod=-2**31 ):
    """CORREL(real0, real1[, timeperiod=?])

    Pearson's Correlation Coefficient (r)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real0) == np.NPY_DOUBLE, "real0 is not double"
    assert real0.ndim == 1, "real0 has wrong dimensions"
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    assert PyArray_TYPE(real1) == np.NPY_DOUBLE, "real1 is not double"
    assert real1.ndim == 1, "real1 has wrong dimensions"
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real0_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_CORREL_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_CORREL( 0 , endidx , <double *>(real0_data+begidx) , <double *>(real1_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def COS( np.ndarray real not None ):
    """COS(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_COS_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_COS( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def COSH( np.ndarray real not None ):
    """COSH(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_COSH_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_COSH( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DEMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """DEMA(real[, timeperiod=?])

    Double Exponential Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_DEMA_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_DEMA( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DIV( np.ndarray real0 not None , np.ndarray real1 not None ):
    """DIV(real0, real1)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real0) == np.NPY_DOUBLE, "real0 is not double"
    assert real0.ndim == 1, "real0 has wrong dimensions"
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    assert PyArray_TYPE(real1) == np.NPY_DOUBLE, "real1 is not double"
    assert real1.ndim == 1, "real1 has wrong dimensions"
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real0_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_DIV_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_DIV( 0 , endidx , <double *>(real0_data+begidx) , <double *>(real1_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def DX( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """DX(high, low, close[, timeperiod=?])

    Directional Movement Index"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_DX_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_DX( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def EMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """EMA(real[, timeperiod=?])

    Exponential Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_EMA_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_EMA( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def EXP( np.ndarray real not None ):
    """EXP(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_EXP_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_EXP( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def FLOOR( np.ndarray real not None ):
    """FLOOR(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_FLOOR_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_FLOOR( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_DCPERIOD( np.ndarray real not None ):
    """HT_DCPERIOD(real)

    Hilbert Transform - Dominant Cycle Period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_HT_DCPERIOD_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_HT_DCPERIOD( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_DCPHASE( np.ndarray real not None ):
    """HT_DCPHASE(real)

    Hilbert Transform - Dominant Cycle Phase"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_HT_DCPHASE_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_HT_DCPHASE( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_PHASOR( np.ndarray real not None ):
    """HT_PHASOR(real)

    Hilbert Transform - Phasor Components"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outinphase
        double* outinphase_data
        np.ndarray outquadrature
        double* outquadrature_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_HT_PHASOR_Lookback( )
    outinphase = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outinphase_data = <double*>outinphase.data
    for i from 0 <= i < min(lookback, length):
        outinphase_data[i] = NaN
    outquadrature = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outquadrature_data = <double*>outquadrature.data
    for i from 0 <= i < min(lookback, length):
        outquadrature_data[i] = NaN
    retCode = TA_HT_PHASOR( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outinphase_data+lookback) , <double *>(outquadrature_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinphase , outquadrature

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_SINE( np.ndarray real not None ):
    """HT_SINE(real)

    Hilbert Transform - SineWave"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outsine
        double* outsine_data
        np.ndarray outleadsine
        double* outleadsine_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_HT_SINE_Lookback( )
    outsine = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outsine_data = <double*>outsine.data
    for i from 0 <= i < min(lookback, length):
        outsine_data[i] = NaN
    outleadsine = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outleadsine_data = <double*>outleadsine.data
    for i from 0 <= i < min(lookback, length):
        outleadsine_data[i] = NaN
    retCode = TA_HT_SINE( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outsine_data+lookback) , <double *>(outleadsine_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outsine , outleadsine

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_TRENDLINE( np.ndarray real not None ):
    """HT_TRENDLINE(real)

    Hilbert Transform - Instantaneous Trendline"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_HT_TRENDLINE_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_HT_TRENDLINE( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def HT_TRENDMODE( np.ndarray real not None ):
    """HT_TRENDMODE(real)

    Hilbert Transform - Trend vs Cycle Mode"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_HT_TRENDMODE_Lookback( )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_HT_TRENDMODE( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def KAMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """KAMA(real[, timeperiod=?])

    Kaufman Adaptive Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_KAMA_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_KAMA( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG( np.ndarray real not None , int timeperiod=-2**31 ):
    """LINEARREG(real[, timeperiod=?])

    Linear Regression"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_LINEARREG_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_LINEARREG( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_ANGLE( np.ndarray real not None , int timeperiod=-2**31 ):
    """LINEARREG_ANGLE(real[, timeperiod=?])

    Linear Regression Angle"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_LINEARREG_ANGLE_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_LINEARREG_ANGLE( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_INTERCEPT( np.ndarray real not None , int timeperiod=-2**31 ):
    """LINEARREG_INTERCEPT(real[, timeperiod=?])

    Linear Regression Intercept"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_LINEARREG_INTERCEPT_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_LINEARREG_INTERCEPT( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LINEARREG_SLOPE( np.ndarray real not None , int timeperiod=-2**31 ):
    """LINEARREG_SLOPE(real[, timeperiod=?])

    Linear Regression Slope"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_LINEARREG_SLOPE_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_LINEARREG_SLOPE( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LN( np.ndarray real not None ):
    """LN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_LN_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_LN( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def LOG10( np.ndarray real not None ):
    """LOG10(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_LOG10_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_LOG10( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MA( np.ndarray real not None , int timeperiod=-2**31 , int matype=0 ):
    """MA(real[, timeperiod=?, matype=?])

    All Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MA_Lookback( timeperiod , matype )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MA( 0 , endidx , <double *>(real_data+begidx) , timeperiod , matype , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACD( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int signalperiod=-2**31 ):
    """MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])

    Moving Average Convergence/Divergence"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        double* outmacd_data
        np.ndarray outmacdsignal
        double* outmacdsignal_data
        np.ndarray outmacdhist
        double* outmacdhist_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MACD_Lookback( fastperiod , slowperiod , signalperiod )
    outmacd = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacd_data = <double*>outmacd.data
    for i from 0 <= i < min(lookback, length):
        outmacd_data[i] = NaN
    outmacdsignal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacdsignal_data = <double*>outmacdsignal.data
    for i from 0 <= i < min(lookback, length):
        outmacdsignal_data[i] = NaN
    outmacdhist = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacdhist_data = <double*>outmacdhist.data
    for i from 0 <= i < min(lookback, length):
        outmacdhist_data[i] = NaN
    retCode = TA_MACD( 0 , endidx , <double *>(real_data+begidx) , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , <double *>(outmacd_data+lookback) , <double *>(outmacdsignal_data+lookback) , <double *>(outmacdhist_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outmacd , outmacdsignal , outmacdhist

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACDEXT( np.ndarray real not None , int fastperiod=-2**31 , int fastmatype=0 , int slowperiod=-2**31 , int slowmatype=0 , int signalperiod=-2**31 , int signalmatype=0 ):
    """MACDEXT(real[, fastperiod=?, fastmatype=?, slowperiod=?, slowmatype=?, signalperiod=?, signalmatype=?])

    MACD with controllable MA type"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        double* outmacd_data
        np.ndarray outmacdsignal
        double* outmacdsignal_data
        np.ndarray outmacdhist
        double* outmacdhist_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MACDEXT_Lookback( fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype )
    outmacd = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacd_data = <double*>outmacd.data
    for i from 0 <= i < min(lookback, length):
        outmacd_data[i] = NaN
    outmacdsignal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacdsignal_data = <double*>outmacdsignal.data
    for i from 0 <= i < min(lookback, length):
        outmacdsignal_data[i] = NaN
    outmacdhist = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacdhist_data = <double*>outmacdhist.data
    for i from 0 <= i < min(lookback, length):
        outmacdhist_data[i] = NaN
    retCode = TA_MACDEXT( 0 , endidx , <double *>(real_data+begidx) , fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype , &outbegidx , &outnbelement , <double *>(outmacd_data+lookback) , <double *>(outmacdsignal_data+lookback) , <double *>(outmacdhist_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outmacd , outmacdsignal , outmacdhist

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MACDFIX( np.ndarray real not None , int signalperiod=-2**31 ):
    """MACDFIX(real[, signalperiod=?])

    Moving Average Convergence/Divergence Fix 12/26"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        double* outmacd_data
        np.ndarray outmacdsignal
        double* outmacdsignal_data
        np.ndarray outmacdhist
        double* outmacdhist_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MACDFIX_Lookback( signalperiod )
    outmacd = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacd_data = <double*>outmacd.data
    for i from 0 <= i < min(lookback, length):
        outmacd_data[i] = NaN
    outmacdsignal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacdsignal_data = <double*>outmacdsignal.data
    for i from 0 <= i < min(lookback, length):
        outmacdsignal_data[i] = NaN
    outmacdhist = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmacdhist_data = <double*>outmacdhist.data
    for i from 0 <= i < min(lookback, length):
        outmacdhist_data[i] = NaN
    retCode = TA_MACDFIX( 0 , endidx , <double *>(real_data+begidx) , signalperiod , &outbegidx , &outnbelement , <double *>(outmacd_data+lookback) , <double *>(outmacdsignal_data+lookback) , <double *>(outmacdhist_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outmacd , outmacdsignal , outmacdhist

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAMA( np.ndarray real not None , double fastlimit=-4e37 , double slowlimit=-4e37 ):
    """MAMA(real[, fastlimit=?, slowlimit=?])

    MESA Adaptive Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outmama
        double* outmama_data
        np.ndarray outfama
        double* outfama_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MAMA_Lookback( fastlimit , slowlimit )
    outmama = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmama_data = <double*>outmama.data
    for i from 0 <= i < min(lookback, length):
        outmama_data[i] = NaN
    outfama = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outfama_data = <double*>outfama.data
    for i from 0 <= i < min(lookback, length):
        outfama_data[i] = NaN
    retCode = TA_MAMA( 0 , endidx , <double *>(real_data+begidx) , fastlimit , slowlimit , &outbegidx , &outnbelement , <double *>(outmama_data+lookback) , <double *>(outfama_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outmama , outfama

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAVP( np.ndarray real not None , np.ndarray periods not None , int minperiod=-2**31 , int maxperiod=-2**31 , int matype=0 ):
    """MAVP(real, periods[, minperiod=?, maxperiod=?, matype=?])"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        double* periods_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    assert PyArray_TYPE(periods) == np.NPY_DOUBLE, "periods is not double"
    assert periods.ndim == 1, "periods has wrong dimensions"
    if not (PyArray_FLAGS(periods) & np.NPY_C_CONTIGUOUS):
        periods = PyArray_GETCONTIGUOUS(periods)
    periods_data = <double*>periods.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MAVP_Lookback( minperiod , maxperiod , matype )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MAVP( 0 , endidx , <double *>(real_data+begidx) , <double *>(periods_data+begidx) , minperiod , maxperiod , matype , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAX( np.ndarray real not None , int timeperiod=-2**31 ):
    """MAX(real[, timeperiod=?])

    Highest value over a specified period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MAX_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MAX( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """MAXINDEX(real[, timeperiod=?])

    Index of highest value over a specified period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MAXINDEX_Lookback( timeperiod )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_MAXINDEX( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MEDPRICE( np.ndarray high not None , np.ndarray low not None ):
    """MEDPRICE(high, low)

    Median Price"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MEDPRICE_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MEDPRICE( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MFI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , np.ndarray volume not None , int timeperiod=-2**31 ):
    """MFI(high, low, close, volume[, timeperiod=?])

    Money Flow Index"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        double* volume_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    assert PyArray_TYPE(volume) == np.NPY_DOUBLE, "volume is not double"
    assert volume.ndim == 1, "volume has wrong dimensions"
    if not (PyArray_FLAGS(volume) & np.NPY_C_CONTIGUOUS):
        volume = PyArray_GETCONTIGUOUS(volume)
    volume_data = <double*>volume.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MFI_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MFI( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , <double *>(volume_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIDPOINT( np.ndarray real not None , int timeperiod=-2**31 ):
    """MIDPOINT(real[, timeperiod=?])

    MidPoint over period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MIDPOINT_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MIDPOINT( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIDPRICE( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """MIDPRICE(high, low[, timeperiod=?])

    Midpoint Price over period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MIDPRICE_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MIDPRICE( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MIN( np.ndarray real not None , int timeperiod=-2**31 ):
    """MIN(real[, timeperiod=?])

    Lowest value over a specified period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MIN_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MIN( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MININDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """MININDEX(real[, timeperiod=?])

    Index of lowest value over a specified period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outinteger
        int* outinteger_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MININDEX_Lookback( timeperiod )
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i from 0 <= i < min(lookback, length):
        outinteger_data[i] = 0
    retCode = TA_MININDEX( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <int *>(outinteger_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outinteger

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINMAX( np.ndarray real not None , int timeperiod=-2**31 ):
    """MINMAX(real[, timeperiod=?])

    Lowest and highest values over a specified period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outmin
        double* outmin_data
        np.ndarray outmax
        double* outmax_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MINMAX_Lookback( timeperiod )
    outmin = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmin_data = <double*>outmin.data
    for i from 0 <= i < min(lookback, length):
        outmin_data[i] = NaN
    outmax = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outmax_data = <double*>outmax.data
    for i from 0 <= i < min(lookback, length):
        outmax_data[i] = NaN
    retCode = TA_MINMAX( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outmin_data+lookback) , <double *>(outmax_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outmin , outmax

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINMAXINDEX( np.ndarray real not None , int timeperiod=-2**31 ):
    """MINMAXINDEX(real[, timeperiod=?])

    Indexes of lowest and highest values over a specified period"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outminidx
        int* outminidx_data
        np.ndarray outmaxidx
        int* outmaxidx_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MINMAXINDEX_Lookback( timeperiod )
    outminidx = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outminidx_data = <int*>outminidx.data
    for i from 0 <= i < min(lookback, length):
        outminidx_data[i] = 0
    outmaxidx = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_DEFAULT)
    outmaxidx_data = <int*>outmaxidx.data
    for i from 0 <= i < min(lookback, length):
        outmaxidx_data[i] = 0
    retCode = TA_MINMAXINDEX( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <int *>(outminidx_data+lookback) , <int *>(outmaxidx_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outminidx , outmaxidx

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """MINUS_DI(high, low, close[, timeperiod=?])

    Minus Directional Indicator"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MINUS_DI_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MINUS_DI( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MINUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """MINUS_DM(high, low[, timeperiod=?])

    Minus Directional Movement"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MINUS_DM_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MINUS_DM( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MOM( np.ndarray real not None , int timeperiod=-2**31 ):
    """MOM(real[, timeperiod=?])

    Momentum"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MOM_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MOM( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def MULT( np.ndarray real0 not None , np.ndarray real1 not None ):
    """MULT(real0, real1)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real0) == np.NPY_DOUBLE, "real0 is not double"
    assert real0.ndim == 1, "real0 has wrong dimensions"
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    assert PyArray_TYPE(real1) == np.NPY_DOUBLE, "real1 is not double"
    assert real1.ndim == 1, "real1 has wrong dimensions"
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real0_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_MULT_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_MULT( 0 , endidx , <double *>(real0_data+begidx) , <double *>(real1_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def NATR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """NATR(high, low, close[, timeperiod=?])

    Normalized Average True Range"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_NATR_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_NATR( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def OBV( np.ndarray real not None , np.ndarray volume not None ):
    """OBV(real, volume)

    On Balance Volume"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        double* volume_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    assert PyArray_TYPE(volume) == np.NPY_DOUBLE, "volume is not double"
    assert volume.ndim == 1, "volume has wrong dimensions"
    if not (PyArray_FLAGS(volume) & np.NPY_C_CONTIGUOUS):
        volume = PyArray_GETCONTIGUOUS(volume)
    volume_data = <double*>volume.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_OBV_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_OBV( 0 , endidx , <double *>(real_data+begidx) , <double *>(volume_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PLUS_DI( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """PLUS_DI(high, low, close[, timeperiod=?])

    Plus Directional Indicator"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_PLUS_DI_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_PLUS_DI( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PLUS_DM( np.ndarray high not None , np.ndarray low not None , int timeperiod=-2**31 ):
    """PLUS_DM(high, low[, timeperiod=?])

    Plus Directional Movement"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_PLUS_DM_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_PLUS_DM( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def PPO( np.ndarray real not None , int fastperiod=-2**31 , int slowperiod=-2**31 , int matype=0 ):
    """PPO(real[, fastperiod=?, slowperiod=?, matype=?])

    Percentage Price Oscillator"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_PPO_Lookback( fastperiod , slowperiod , matype )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_PPO( 0 , endidx , <double *>(real_data+begidx) , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROC( np.ndarray real not None , int timeperiod=-2**31 ):
    """ROC(real[, timeperiod=?])

    Rate of change : ((price/prevPrice)-1)*100"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ROC_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ROC( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCP( np.ndarray real not None , int timeperiod=-2**31 ):
    """ROCP(real[, timeperiod=?])

    Rate of change Percentage: (price-prevPrice)/prevPrice"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ROCP_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ROCP( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCR( np.ndarray real not None , int timeperiod=-2**31 ):
    """ROCR(real[, timeperiod=?])

    Rate of change ratio: (price/prevPrice)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ROCR_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ROCR( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ROCR100( np.ndarray real not None , int timeperiod=-2**31 ):
    """ROCR100(real[, timeperiod=?])

    Rate of change ratio 100 scale: (price/prevPrice)*100"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ROCR100_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ROCR100( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def RSI( np.ndarray real not None , int timeperiod=-2**31 ):
    """RSI(real[, timeperiod=?])

    Relative Strength Index"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_RSI_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_RSI( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SAR( np.ndarray high not None , np.ndarray low not None , double acceleration=-4e37 , double maximum=-4e37 ):
    """SAR(high, low[, acceleration=?, maximum=?])

    Parabolic SAR"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_SAR_Lookback( acceleration , maximum )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_SAR( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , acceleration , maximum , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SAREXT( np.ndarray high not None , np.ndarray low not None , double startvalue=-4e37 , double offsetonreverse=-4e37 , double accelerationinitlong=-4e37 , double accelerationlong=-4e37 , double accelerationmaxlong=-4e37 , double accelerationinitshort=-4e37 , double accelerationshort=-4e37 , double accelerationmaxshort=-4e37 ):
    """SAREXT(high, low[, startvalue=?, offsetonreverse=?, accelerationinitlong=?, accelerationlong=?, accelerationmaxlong=?, accelerationinitshort=?, accelerationshort=?, accelerationmaxshort=?])

    Parabolic SAR - Extended"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_SAREXT_Lookback( startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_SAREXT( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SIN( np.ndarray real not None ):
    """SIN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_SIN_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_SIN( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SINH( np.ndarray real not None ):
    """SINH(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_SINH_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_SINH( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """SMA(real[, timeperiod=?])

    Simple Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_SMA_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_SMA( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SQRT( np.ndarray real not None ):
    """SQRT(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_SQRT_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_SQRT( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STDDEV( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """STDDEV(real[, timeperiod=?, nbdev=?])

    Standard Deviation"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_STDDEV_Lookback( timeperiod , nbdev )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_STDDEV( 0 , endidx , <double *>(real_data+begidx) , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCH( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int slowk_period=-2**31 , int slowk_matype=0 , int slowd_period=-2**31 , int slowd_matype=0 ):
    """STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])

    Stochastic"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outslowk
        double* outslowk_data
        np.ndarray outslowd
        double* outslowd_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_STOCH_Lookback( fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype )
    outslowk = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outslowk_data = <double*>outslowk.data
    for i from 0 <= i < min(lookback, length):
        outslowk_data[i] = NaN
    outslowd = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outslowd_data = <double*>outslowd.data
    for i from 0 <= i < min(lookback, length):
        outslowd_data[i] = NaN
    retCode = TA_STOCH( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , <double *>(outslowk_data+lookback) , <double *>(outslowd_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outslowk , outslowd

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCHF( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
    """STOCHF(high, low, close[, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Fast"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outfastk
        double* outfastk_data
        np.ndarray outfastd
        double* outfastd_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_STOCHF_Lookback( fastk_period , fastd_period , fastd_matype )
    outfastk = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outfastk_data = <double*>outfastk.data
    for i from 0 <= i < min(lookback, length):
        outfastk_data[i] = NaN
    outfastd = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outfastd_data = <double*>outfastd.data
    for i from 0 <= i < min(lookback, length):
        outfastd_data[i] = NaN
    retCode = TA_STOCHF( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>(outfastk_data+lookback) , <double *>(outfastd_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outfastk , outfastd

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def STOCHRSI( np.ndarray real not None , int timeperiod=-2**31 , int fastk_period=-2**31 , int fastd_period=-2**31 , int fastd_matype=0 ):
    """STOCHRSI(real[, timeperiod=?, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Relative Strength Index"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outfastk
        double* outfastk_data
        np.ndarray outfastd
        double* outfastd_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_STOCHRSI_Lookback( timeperiod , fastk_period , fastd_period , fastd_matype )
    outfastk = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outfastk_data = <double*>outfastk.data
    for i from 0 <= i < min(lookback, length):
        outfastk_data[i] = NaN
    outfastd = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outfastd_data = <double*>outfastd.data
    for i from 0 <= i < min(lookback, length):
        outfastd_data[i] = NaN
    retCode = TA_STOCHRSI( 0 , endidx , <double *>(real_data+begidx) , timeperiod , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>(outfastk_data+lookback) , <double *>(outfastd_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outfastk , outfastd

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SUB( np.ndarray real0 not None , np.ndarray real1 not None ):
    """SUB(real0, real1)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real0_data
        double* real1_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real0) == np.NPY_DOUBLE, "real0 is not double"
    assert real0.ndim == 1, "real0 has wrong dimensions"
    if not (PyArray_FLAGS(real0) & np.NPY_C_CONTIGUOUS):
        real0 = PyArray_GETCONTIGUOUS(real0)
    real0_data = <double*>real0.data
    assert PyArray_TYPE(real1) == np.NPY_DOUBLE, "real1 is not double"
    assert real1.ndim == 1, "real1 has wrong dimensions"
    if not (PyArray_FLAGS(real1) & np.NPY_C_CONTIGUOUS):
        real1 = PyArray_GETCONTIGUOUS(real1)
    real1_data = <double*>real1.data
    length = real0.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real0_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_SUB_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_SUB( 0 , endidx , <double *>(real0_data+begidx) , <double *>(real1_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def SUM( np.ndarray real not None , int timeperiod=-2**31 ):
    """SUM(real[, timeperiod=?])

    Summation"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_SUM_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_SUM( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def T3( np.ndarray real not None , int timeperiod=-2**31 , double vfactor=-4e37 ):
    """T3(real[, timeperiod=?, vfactor=?])

    Triple Exponential Moving Average (T3)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_T3_Lookback( timeperiod , vfactor )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_T3( 0 , endidx , <double *>(real_data+begidx) , timeperiod , vfactor , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TAN( np.ndarray real not None ):
    """TAN(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_TAN_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_TAN( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TANH( np.ndarray real not None ):
    """TANH(real)"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_TANH_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_TANH( 0 , endidx , <double *>(real_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TEMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """TEMA(real[, timeperiod=?])

    Triple Exponential Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_TEMA_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_TEMA( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TRANGE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """TRANGE(high, low, close)

    True Range"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_TRANGE_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_TRANGE( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TRIMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """TRIMA(real[, timeperiod=?])

    Triangular Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_TRIMA_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_TRIMA( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TRIX( np.ndarray real not None , int timeperiod=-2**31 ):
    """TRIX(real[, timeperiod=?])

    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_TRIX_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_TRIX( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TSF( np.ndarray real not None , int timeperiod=-2**31 ):
    """TSF(real[, timeperiod=?])

    Time Series Forecast"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_TSF_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_TSF( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def TYPPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """TYPPRICE(high, low, close)

    Typical Price"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_TYPPRICE_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_TYPPRICE( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def ULTOSC( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod1=-2**31 , int timeperiod2=-2**31 , int timeperiod3=-2**31 ):
    """ULTOSC(high, low, close[, timeperiod1=?, timeperiod2=?, timeperiod3=?])

    Ultimate Oscillator"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_ULTOSC_Lookback( timeperiod1 , timeperiod2 , timeperiod3 )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_ULTOSC( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod1 , timeperiod2 , timeperiod3 , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def VAR( np.ndarray real not None , int timeperiod=-2**31 , double nbdev=-4e37 ):
    """VAR(real[, timeperiod=?, nbdev=?])

    Variance"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_VAR_Lookback( timeperiod , nbdev )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_VAR( 0 , endidx , <double *>(real_data+begidx) , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WCLPRICE( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None ):
    """WCLPRICE(high, low, close)

    Weighted Close Price"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_WCLPRICE_Lookback( )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_WCLPRICE( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WILLR( np.ndarray high not None , np.ndarray low not None , np.ndarray close not None , int timeperiod=-2**31 ):
    """WILLR(high, low, close[, timeperiod=?])

    Williams' %R"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(high) == np.NPY_DOUBLE, "high is not double"
    assert high.ndim == 1, "high has wrong dimensions"
    if not (PyArray_FLAGS(high) & np.NPY_C_CONTIGUOUS):
        high = PyArray_GETCONTIGUOUS(high)
    high_data = <double*>high.data
    assert PyArray_TYPE(low) == np.NPY_DOUBLE, "low is not double"
    assert low.ndim == 1, "low has wrong dimensions"
    if not (PyArray_FLAGS(low) & np.NPY_C_CONTIGUOUS):
        low = PyArray_GETCONTIGUOUS(low)
    low_data = <double*>low.data
    assert PyArray_TYPE(close) == np.NPY_DOUBLE, "close is not double"
    assert close.ndim == 1, "close has wrong dimensions"
    if not (PyArray_FLAGS(close) & np.NPY_C_CONTIGUOUS):
        close = PyArray_GETCONTIGUOUS(close)
    close_data = <double*>close.data
    length = high.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(high_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_WILLR_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_WILLR( 0 , endidx , <double *>(high_data+begidx) , <double *>(low_data+begidx) , <double *>(close_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
def WMA( np.ndarray real not None , int timeperiod=-2**31 ):
    """WMA(real[, timeperiod=?])

    Weighted Moving Average"""
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
        double* outreal_data
    assert PyArray_TYPE(real) == np.NPY_DOUBLE, "real is not double"
    assert real.ndim == 1, "real has wrong dimensions"
    if not (PyArray_FLAGS(real) & np.NPY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    real_data = <double*>real.data
    length = real.shape[0]
    begidx = 0
    for i from 0 <= i < length:
        if not isnan(real_data[i]):
            begidx = i
            break
    else:
        raise Exception("inputs are all NaN")
    endidx = length - begidx - 1
    TA_Initialize()
    lookback = begidx + TA_WMA_Lookback( timeperiod )
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i from 0 <= i < min(lookback, length):
        outreal_data[i] = NaN
    retCode = TA_WMA( 0 , endidx , <double *>(real_data+begidx) , timeperiod , &outbegidx , &outnbelement , <double *>(outreal_data+lookback) )
    TA_Shutdown()
    if retCode != TA_SUCCESS:
        raise Exception("%d: %s" % (retCode, RetCodes.get(retCode, "Unknown")))
    return outreal

__all__ = ["ACOS","AD","ADD","ADOSC","ADX","ADXR","APO","AROON","AROONOSC","ASIN","ATAN","ATR","AVGPRICE","BBANDS","BETA","BOP","CCI","CDL2CROWS","CDL3BLACKCROWS","CDL3INSIDE","CDL3LINESTRIKE","CDL3OUTSIDE","CDL3STARSINSOUTH","CDL3WHITESOLDIERS","CDLABANDONEDBABY","CDLADVANCEBLOCK","CDLBELTHOLD","CDLBREAKAWAY","CDLCLOSINGMARUBOZU","CDLCONCEALBABYSWALL","CDLCOUNTERATTACK","CDLDARKCLOUDCOVER","CDLDOJI","CDLDOJISTAR","CDLDRAGONFLYDOJI","CDLENGULFING","CDLEVENINGDOJISTAR","CDLEVENINGSTAR","CDLGAPSIDESIDEWHITE","CDLGRAVESTONEDOJI","CDLHAMMER","CDLHANGINGMAN","CDLHARAMI","CDLHARAMICROSS","CDLHIGHWAVE","CDLHIKKAKE","CDLHIKKAKEMOD","CDLHOMINGPIGEON","CDLIDENTICAL3CROWS","CDLINNECK","CDLINVERTEDHAMMER","CDLKICKING","CDLKICKINGBYLENGTH","CDLLADDERBOTTOM","CDLLONGLEGGEDDOJI","CDLLONGLINE","CDLMARUBOZU","CDLMATCHINGLOW","CDLMATHOLD","CDLMORNINGDOJISTAR","CDLMORNINGSTAR","CDLONNECK","CDLPIERCING","CDLRICKSHAWMAN","CDLRISEFALL3METHODS","CDLSEPARATINGLINES","CDLSHOOTINGSTAR","CDLSHORTLINE","CDLSPINNINGTOP","CDLSTALLEDPATTERN","CDLSTICKSANDWICH","CDLTAKURI","CDLTASUKIGAP","CDLTHRUSTING","CDLTRISTAR","CDLUNIQUE3RIVER","CDLUPSIDEGAP2CROWS","CDLXSIDEGAP3METHODS","CEIL","CMO","CORREL","COS","COSH","DEMA","DIV","DX","EMA","EXP","FLOOR","HT_DCPERIOD","HT_DCPHASE","HT_PHASOR","HT_SINE","HT_TRENDLINE","HT_TRENDMODE","KAMA","LINEARREG","LINEARREG_ANGLE","LINEARREG_INTERCEPT","LINEARREG_SLOPE","LN","LOG10","MA","MACD","MACDEXT","MACDFIX","MAMA","MAVP","MAX","MAXINDEX","MEDPRICE","MFI","MIDPOINT","MIDPRICE","MIN","MININDEX","MINMAX","MINMAXINDEX","MINUS_DI","MINUS_DM","MOM","MULT","NATR","OBV","PLUS_DI","PLUS_DM","PPO","ROC","ROCP","ROCR","ROCR100","RSI","SAR","SAREXT","SIN","SINH","SMA","SQRT","STDDEV","STOCH","STOCHF","STOCHRSI","SUB","SUM","T3","TAN","TANH","TEMA","TRANGE","TRIMA","TRIX","TSF","TYPPRICE","ULTOSC","VAR","WCLPRICE","WILLR","WMA"]
