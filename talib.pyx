
import numpy
cimport numpy as np

ctypedef int TA_RetCode
ctypedef int TA_MAType

# TA_MAType enums
SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3 = range(9)

RetCodes = {
  0 : "Success",
  1 : "LibNotInitialize",
  2 : "BadParam",
  3 : "AllocErr",
  4 : "GroupNotFound",
  5 : "FuncNotFound",
  6 : "InvalidHandle",
  7 : "InvalidParamHolder",
  8 : "InvalidParamHolderType",
  9 : "InvalidParamFunction",
  10 : "InputNotAllInitialize",
  11 : "OutputNotAllInitialize",
  12 : "OutOfRangeStartIndex",
  13 : "OutOfRangeEndIndex",
  14 : "InvalidListType",
  15 : "BadObject",
  16 : "NotSupported",
  5000 : "InternalError",
  0xFFFF : "UnknownErr",
}

# extract the needed part of ta_libc.h that I will use in the interface
cdef extern from "ta_libc.h":
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

def acos( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_ACOS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ACOS( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ad( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , np.ndarray[np.float_t, ndim=1] volume ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_AD_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_AD( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , <double *>volume.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def add( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 ):
    cdef int startidx = 0
    cdef int endidx = real0.shape[0] - 1
    cdef int lookback = TA_ADD_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ADD( startidx , endidx , <double *>real0.data , <double *>real1.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def adosc( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , np.ndarray[np.float_t, ndim=1] volume , fastperiod=-2**31 , slowperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_ADOSC_Lookback( fastperiod , slowperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ADOSC( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , <double *>volume.data , fastperiod , slowperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def adx( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_ADX_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ADX( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def adxr( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_ADXR_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ADXR( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def apo( np.ndarray[np.float_t, ndim=1] real , fastperiod=-2**31 , slowperiod=-2**31 , matype=0 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_APO_Lookback( fastperiod , slowperiod , matype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_APO( startidx , endidx , <double *>real.data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def aroon( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_AROON_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outaroondown = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outaroonup = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_AROON( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outaroondown.data , <double *>outaroonup.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outaroondown , outaroonup )

def aroonosc( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_AROONOSC_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_AROONOSC( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def asin( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_ASIN_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ASIN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def atan( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_ATAN_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ATAN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def atr( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_ATR_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ATR( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def avgprice( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_AVGPRICE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_AVGPRICE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def bbands( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , nbdevup=-4e37 , nbdevdn=-4e37 , matype=0 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_BBANDS_Lookback( timeperiod , nbdevup , nbdevdn , matype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outrealupperband = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outrealmiddleband = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outreallowerband = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_BBANDS( startidx , endidx , <double *>real.data , timeperiod , nbdevup , nbdevdn , matype , &outbegidx , &outnbelement , <double *>outrealupperband.data , <double *>outrealmiddleband.data , <double *>outreallowerband.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outrealupperband , outrealmiddleband , outreallowerband )

def beta( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real0.shape[0] - 1
    cdef int lookback = TA_BETA_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_BETA( startidx , endidx , <double *>real0.data , <double *>real1.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def bop( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_BOP_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_BOP( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def cci( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CCI_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CCI( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def cdl2crows( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDL2CROWS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDL2CROWS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdl3blackcrows( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDL3BLACKCROWS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDL3BLACKCROWS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdl3inside( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDL3INSIDE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDL3INSIDE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdl3linestrike( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDL3LINESTRIKE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDL3LINESTRIKE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdl3outside( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDL3OUTSIDE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDL3OUTSIDE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdl3starsinsouth( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDL3STARSINSOUTH_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDL3STARSINSOUTH( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdl3whitesoldiers( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDL3WHITESOLDIERS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDL3WHITESOLDIERS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlabandonedbaby( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLABANDONEDBABY_Lookback( penetration )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLABANDONEDBABY( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdladvanceblock( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLADVANCEBLOCK_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLADVANCEBLOCK( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlbelthold( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLBELTHOLD_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLBELTHOLD( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlbreakaway( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLBREAKAWAY_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLBREAKAWAY( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlclosingmarubozu( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLCLOSINGMARUBOZU_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLCLOSINGMARUBOZU( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlconcealbabyswall( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLCONCEALBABYSWALL_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLCONCEALBABYSWALL( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlcounterattack( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLCOUNTERATTACK_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLCOUNTERATTACK( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdldarkcloudcover( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLDARKCLOUDCOVER_Lookback( penetration )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLDARKCLOUDCOVER( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdldoji( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLDOJI_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLDOJI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdldojistar( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLDOJISTAR_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLDOJISTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdldragonflydoji( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLDRAGONFLYDOJI_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLDRAGONFLYDOJI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlengulfing( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLENGULFING_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLENGULFING( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdleveningdojistar( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLEVENINGDOJISTAR_Lookback( penetration )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLEVENINGDOJISTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdleveningstar( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLEVENINGSTAR_Lookback( penetration )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLEVENINGSTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlgapsidesidewhite( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLGAPSIDESIDEWHITE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLGAPSIDESIDEWHITE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlgravestonedoji( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLGRAVESTONEDOJI_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLGRAVESTONEDOJI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlhammer( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLHAMMER_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLHAMMER( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlhangingman( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLHANGINGMAN_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLHANGINGMAN( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlharami( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLHARAMI_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLHARAMI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlharamicross( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLHARAMICROSS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLHARAMICROSS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlhighwave( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLHIGHWAVE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLHIGHWAVE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlhikkake( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLHIKKAKE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLHIKKAKE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlhikkakemod( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLHIKKAKEMOD_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLHIKKAKEMOD( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlhomingpigeon( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLHOMINGPIGEON_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLHOMINGPIGEON( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlidentical3crows( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLIDENTICAL3CROWS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLIDENTICAL3CROWS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlinneck( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLINNECK_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLINNECK( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlinvertedhammer( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLINVERTEDHAMMER_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLINVERTEDHAMMER( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlkicking( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLKICKING_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLKICKING( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlkickingbylength( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLKICKINGBYLENGTH_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLKICKINGBYLENGTH( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlladderbottom( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLLADDERBOTTOM_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLLADDERBOTTOM( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdllongleggeddoji( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLLONGLEGGEDDOJI_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLLONGLEGGEDDOJI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdllongline( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLLONGLINE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLLONGLINE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlmarubozu( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLMARUBOZU_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLMARUBOZU( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlmatchinglow( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLMATCHINGLOW_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLMATCHINGLOW( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlmathold( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLMATHOLD_Lookback( penetration )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLMATHOLD( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlmorningdojistar( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLMORNINGDOJISTAR_Lookback( penetration )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLMORNINGDOJISTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlmorningstar( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLMORNINGSTAR_Lookback( penetration )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLMORNINGSTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlonneck( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLONNECK_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLONNECK( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlpiercing( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLPIERCING_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLPIERCING( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlrickshawman( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLRICKSHAWMAN_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLRICKSHAWMAN( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlrisefall3methods( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLRISEFALL3METHODS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLRISEFALL3METHODS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlseparatinglines( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLSEPARATINGLINES_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLSEPARATINGLINES( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlshootingstar( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLSHOOTINGSTAR_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLSHOOTINGSTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlshortline( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLSHORTLINE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLSHORTLINE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlspinningtop( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLSPINNINGTOP_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLSPINNINGTOP( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlstalledpattern( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLSTALLEDPATTERN_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLSTALLEDPATTERN( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlsticksandwich( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLSTICKSANDWICH_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLSTICKSANDWICH( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdltakuri( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLTAKURI_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLTAKURI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdltasukigap( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLTASUKIGAP_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLTASUKIGAP( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlthrusting( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLTHRUSTING_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLTHRUSTING( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdltristar( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLTRISTAR_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLTRISTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlunique3river( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLUNIQUE3RIVER_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLUNIQUE3RIVER( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlupsidegap2crows( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLUPSIDEGAP2CROWS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLUPSIDEGAP2CROWS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def cdlxsidegap3methods( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_CDLXSIDEGAP3METHODS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CDLXSIDEGAP3METHODS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def ceil( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_CEIL_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CEIL( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def cmo( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_CMO_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CMO( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def correl( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real0.shape[0] - 1
    cdef int lookback = TA_CORREL_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_CORREL( startidx , endidx , <double *>real0.data , <double *>real1.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def cos( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_COS_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_COS( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def cosh( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_COSH_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_COSH( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def dema( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_DEMA_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_DEMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def div( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 ):
    cdef int startidx = 0
    cdef int endidx = real0.shape[0] - 1
    cdef int lookback = TA_DIV_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_DIV( startidx , endidx , <double *>real0.data , <double *>real1.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def dx( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_DX_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_DX( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ema( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_EMA_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_EMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def exp( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_EXP_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_EXP( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def floor( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_FLOOR_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_FLOOR( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ht_dcperiod( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_HT_DCPERIOD_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_HT_DCPERIOD( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ht_dcphase( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_HT_DCPHASE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_HT_DCPHASE( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ht_phasor( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_HT_PHASOR_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outinphase = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outquadrature = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_HT_PHASOR( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outinphase.data , <double *>outquadrature.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinphase , outquadrature )

def ht_sine( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_HT_SINE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outsine = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outleadsine = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_HT_SINE( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outsine.data , <double *>outleadsine.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outsine , outleadsine )

def ht_trendline( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_HT_TRENDLINE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_HT_TRENDLINE( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ht_trendmode( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_HT_TRENDMODE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_HT_TRENDMODE( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def kama( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_KAMA_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_KAMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def linearreg( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_LINEARREG_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_LINEARREG( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def linearreg_angle( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_LINEARREG_ANGLE_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_LINEARREG_ANGLE( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def linearreg_intercept( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_LINEARREG_INTERCEPT_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_LINEARREG_INTERCEPT( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def linearreg_slope( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_LINEARREG_SLOPE_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_LINEARREG_SLOPE( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ln( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_LN_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_LN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def log10( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_LOG10_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_LOG10( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ma( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , matype=0 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MA_Lookback( timeperiod , matype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MA( startidx , endidx , <double *>real.data , timeperiod , matype , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def macd( np.ndarray[np.float_t, ndim=1] real , fastperiod=-2**31 , slowperiod=-2**31 , signalperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MACD_Lookback( fastperiod , slowperiod , signalperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outmacd = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outmacdsignal = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outmacdhist = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MACD( startidx , endidx , <double *>real.data , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , <double *>outmacd.data , <double *>outmacdsignal.data , <double *>outmacdhist.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outmacd , outmacdsignal , outmacdhist )

def macdext( np.ndarray[np.float_t, ndim=1] real , fastperiod=-2**31 , fastmatype=0 , slowperiod=-2**31 , slowmatype=0 , signalperiod=-2**31 , signalmatype=0 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MACDEXT_Lookback( fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outmacd = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outmacdsignal = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outmacdhist = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MACDEXT( startidx , endidx , <double *>real.data , fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype , &outbegidx , &outnbelement , <double *>outmacd.data , <double *>outmacdsignal.data , <double *>outmacdhist.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outmacd , outmacdsignal , outmacdhist )

def macdfix( np.ndarray[np.float_t, ndim=1] real , signalperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MACDFIX_Lookback( signalperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outmacd = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outmacdsignal = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outmacdhist = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MACDFIX( startidx , endidx , <double *>real.data , signalperiod , &outbegidx , &outnbelement , <double *>outmacd.data , <double *>outmacdsignal.data , <double *>outmacdhist.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outmacd , outmacdsignal , outmacdhist )

def mama( np.ndarray[np.float_t, ndim=1] real , fastlimit=-4e37 , slowlimit=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MAMA_Lookback( fastlimit , slowlimit )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outmama = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outfama = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MAMA( startidx , endidx , <double *>real.data , fastlimit , slowlimit , &outbegidx , &outnbelement , <double *>outmama.data , <double *>outfama.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outmama , outfama )

def mavp( np.ndarray[np.float_t, ndim=1] real , np.ndarray[np.float_t, ndim=1] periods , minperiod=-2**31 , maxperiod=-2**31 , matype=0 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MAVP_Lookback( minperiod , maxperiod , matype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MAVP( startidx , endidx , <double *>real.data , <double *>periods.data , minperiod , maxperiod , matype , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def max( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MAX_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MAX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def maxindex( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MAXINDEX_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MAXINDEX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def medprice( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_MEDPRICE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MEDPRICE( startidx , endidx , <double *>high.data , <double *>low.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def mfi( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , np.ndarray[np.float_t, ndim=1] volume , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_MFI_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MFI( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , <double *>volume.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def midpoint( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MIDPOINT_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MIDPOINT( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def midprice( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_MIDPRICE_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MIDPRICE( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def min( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MIN_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MIN( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def minindex( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MININDEX_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outinteger = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MININDEX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outinteger )

def minmax( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MINMAX_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outmin = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outmax = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MINMAX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outmin.data , <double *>outmax.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outmin , outmax )

def minmaxindex( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MINMAXINDEX_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.int_t, ndim=1] outminidx = numpy.zeros(allocation)
    cdef np.ndarray[np.int_t, ndim=1] outmaxidx = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MINMAXINDEX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <int *>outminidx.data , <int *>outmaxidx.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outminidx , outmaxidx )

def minus_di( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_MINUS_DI_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MINUS_DI( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def minus_dm( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_MINUS_DM_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MINUS_DM( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def mom( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_MOM_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MOM( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def mult( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 ):
    cdef int startidx = 0
    cdef int endidx = real0.shape[0] - 1
    cdef int lookback = TA_MULT_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_MULT( startidx , endidx , <double *>real0.data , <double *>real1.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def natr( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_NATR_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_NATR( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def obv( np.ndarray[np.float_t, ndim=1] real , np.ndarray[np.float_t, ndim=1] volume ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_OBV_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_OBV( startidx , endidx , <double *>real.data , <double *>volume.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def plus_di( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_PLUS_DI_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_PLUS_DI( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def plus_dm( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_PLUS_DM_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_PLUS_DM( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ppo( np.ndarray[np.float_t, ndim=1] real , fastperiod=-2**31 , slowperiod=-2**31 , matype=0 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_PPO_Lookback( fastperiod , slowperiod , matype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_PPO( startidx , endidx , <double *>real.data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def roc( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_ROC_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ROC( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def rocp( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_ROCP_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ROCP( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def rocr( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_ROCR_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ROCR( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def rocr100( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_ROCR100_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ROCR100( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def rsi( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_RSI_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_RSI( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def sar( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , acceleration=-4e37 , maximum=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_SAR_Lookback( acceleration , maximum )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_SAR( startidx , endidx , <double *>high.data , <double *>low.data , acceleration , maximum , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def sarext( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , startvalue=-4e37 , offsetonreverse=-4e37 , accelerationinitlong=-4e37 , accelerationlong=-4e37 , accelerationmaxlong=-4e37 , accelerationinitshort=-4e37 , accelerationshort=-4e37 , accelerationmaxshort=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_SAREXT_Lookback( startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_SAREXT( startidx , endidx , <double *>high.data , <double *>low.data , startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def sin( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_SIN_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_SIN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def sinh( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_SINH_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_SINH( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def sma( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_SMA_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_SMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def sqrt( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_SQRT_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_SQRT( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def stddev( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , nbdev=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_STDDEV_Lookback( timeperiod , nbdev )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_STDDEV( startidx , endidx , <double *>real.data , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def stoch( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , fastk_period=-2**31 , slowk_period=-2**31 , slowk_matype=0 , slowd_period=-2**31 , slowd_matype=0 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_STOCH_Lookback( fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outslowk = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outslowd = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_STOCH( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , <double *>outslowk.data , <double *>outslowd.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outslowk , outslowd )

def stochf( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , fastk_period=-2**31 , fastd_period=-2**31 , fastd_matype=0 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_STOCHF_Lookback( fastk_period , fastd_period , fastd_matype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outfastk = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outfastd = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_STOCHF( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>outfastk.data , <double *>outfastd.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outfastk , outfastd )

def stochrsi( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , fastk_period=-2**31 , fastd_period=-2**31 , fastd_matype=0 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_STOCHRSI_Lookback( timeperiod , fastk_period , fastd_period , fastd_matype )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outfastk = numpy.zeros(allocation)
    cdef np.ndarray[np.float_t, ndim=1] outfastd = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_STOCHRSI( startidx , endidx , <double *>real.data , timeperiod , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>outfastk.data , <double *>outfastd.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outfastk , outfastd )

def sub( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 ):
    cdef int startidx = 0
    cdef int endidx = real0.shape[0] - 1
    cdef int lookback = TA_SUB_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_SUB( startidx , endidx , <double *>real0.data , <double *>real1.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def sum( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_SUM_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_SUM( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def t3( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , vfactor=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_T3_Lookback( timeperiod , vfactor )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_T3( startidx , endidx , <double *>real.data , timeperiod , vfactor , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def tan( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_TAN_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_TAN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def tanh( np.ndarray[np.float_t, ndim=1] real ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_TANH_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_TANH( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def tema( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_TEMA_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_TEMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def trange( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_TRANGE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_TRANGE( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def trima( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_TRIMA_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_TRIMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def trix( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_TRIX_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_TRIX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def tsf( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_TSF_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_TSF( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def typprice( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_TYPPRICE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_TYPPRICE( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def ultosc( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod1=-2**31 , timeperiod2=-2**31 , timeperiod3=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_ULTOSC_Lookback( timeperiod1 , timeperiod2 , timeperiod3 )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_ULTOSC( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod1 , timeperiod2 , timeperiod3 , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def var( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , nbdev=-4e37 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_VAR_Lookback( timeperiod , nbdev )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_VAR( startidx , endidx , <double *>real.data , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def wclprice( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_WCLPRICE_Lookback( )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_WCLPRICE( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def willr( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = high.shape[0] - 1
    cdef int lookback = TA_WILLR_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_WILLR( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

def wma( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    cdef int startidx = 0
    cdef int endidx = real.shape[0] - 1
    cdef int lookback = TA_WMA_Lookback( timeperiod )
    cdef int temp = max(lookback, startidx )
    cdef int allocation
    if ( temp > endidx ):
        allocation = 0
    else:
        allocation = endidx - temp + 1
    cdef int outbegidx
    cdef int outnbelement
    cdef np.ndarray[np.float_t, ndim=1] outreal = numpy.zeros(allocation)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    retCode = TA_WMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%s (%d)" % (RetCodes[retCode], retCode))
    TA_Shutdown()
    return ( outbegidx , outnbelement , outreal )

__all__ = [acos,ad,add,adosc,adx,adxr,apo,aroon,aroonosc,asin,atan,atr,avgprice,bbands,beta,bop,cci,cdl2crows,cdl3blackcrows,cdl3inside,cdl3linestrike,cdl3outside,cdl3starsinsouth,cdl3whitesoldiers,cdlabandonedbaby,cdladvanceblock,cdlbelthold,cdlbreakaway,cdlclosingmarubozu,cdlconcealbabyswall,cdlcounterattack,cdldarkcloudcover,cdldoji,cdldojistar,cdldragonflydoji,cdlengulfing,cdleveningdojistar,cdleveningstar,cdlgapsidesidewhite,cdlgravestonedoji,cdlhammer,cdlhangingman,cdlharami,cdlharamicross,cdlhighwave,cdlhikkake,cdlhikkakemod,cdlhomingpigeon,cdlidentical3crows,cdlinneck,cdlinvertedhammer,cdlkicking,cdlkickingbylength,cdlladderbottom,cdllongleggeddoji,cdllongline,cdlmarubozu,cdlmatchinglow,cdlmathold,cdlmorningdojistar,cdlmorningstar,cdlonneck,cdlpiercing,cdlrickshawman,cdlrisefall3methods,cdlseparatinglines,cdlshootingstar,cdlshortline,cdlspinningtop,cdlstalledpattern,cdlsticksandwich,cdltakuri,cdltasukigap,cdlthrusting,cdltristar,cdlunique3river,cdlupsidegap2crows,cdlxsidegap3methods,ceil,cmo,correl,cos,cosh,dema,div,dx,ema,exp,floor,ht_dcperiod,ht_dcphase,ht_phasor,ht_sine,ht_trendline,ht_trendmode,kama,linearreg,linearreg_angle,linearreg_intercept,linearreg_slope,ln,log10,ma,macd,macdext,macdfix,mama,mavp,max,maxindex,medprice,mfi,midpoint,midprice,min,minindex,minmax,minmaxindex,minus_di,minus_dm,mom,mult,natr,obv,plus_di,plus_dm,ppo,roc,rocp,rocr,rocr100,rsi,sar,sarext,sin,sinh,sma,sqrt,stddev,stoch,stochf,stochrsi,sub,sum,t3,tan,tanh,tema,trange,trima,trix,tsf,typprice,ultosc,var,wclprice,willr,wma]
