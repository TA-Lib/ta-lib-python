
import numpy
cimport numpy as np

ctypedef int TA_RetCode
ctypedef int TA_MAType

# TA_MAType enums
MA_SMA, MA_EMA, MA_WMA, MA_DEMA, MA_TEMA, MA_TRIMA, MA_KAMA, MA_MAMA, MA_T3 = range(9)

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

def ACOS( np.ndarray[np.float_t, ndim=1] real ):
    """ACOS(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ACOS( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def AD( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , np.ndarray[np.float_t, ndim=1] volume ):
    """AD(high, low, close, volume)

    Chaikin A/D Line"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_AD( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , <double *>volume.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ADD( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 ):
    """ADD(real0, real1)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ADD( startidx , endidx , <double *>real0.data , <double *>real1.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ADOSC( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , np.ndarray[np.float_t, ndim=1] volume , fastperiod=-2**31 , slowperiod=-2**31 ):
    """ADOSC(high, low, close, volume[, fastperiod=?, slowperiod=?])

    Chaikin A/D Oscillator"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ADOSC( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , <double *>volume.data , fastperiod , slowperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ADX( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """ADX(high, low, close[, timeperiod=?])

    Average Directional Movement Index"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ADX( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ADXR( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """ADXR(high, low, close[, timeperiod=?])

    Average Directional Movement Index Rating"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ADXR( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def APO( np.ndarray[np.float_t, ndim=1] real , fastperiod=-2**31 , slowperiod=-2**31 , matype=0 ):
    """APO(real[, fastperiod=?, slowperiod=?, matype=?])

    Absolute Price Oscillator"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_APO( startidx , endidx , <double *>real.data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def AROON( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    """AROON(high, low[, timeperiod=?])

    Aroon"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outaroondown = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outaroonup = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_AROON( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outaroondown.data , <double *>outaroonup.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outaroondown , outaroonup )

def AROONOSC( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    """AROONOSC(high, low[, timeperiod=?])

    Aroon Oscillator"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_AROONOSC( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ASIN( np.ndarray[np.float_t, ndim=1] real ):
    """ASIN(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ASIN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ATAN( np.ndarray[np.float_t, ndim=1] real ):
    """ATAN(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ATAN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ATR( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """ATR(high, low, close[, timeperiod=?])

    Average True Range"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ATR( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def AVGPRICE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """AVGPRICE(open, high, low, close)

    Average Price"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_AVGPRICE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def BBANDS( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , nbdevup=-4e37 , nbdevdn=-4e37 , matype=0 ):
    """BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])

    Bollinger Bands"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outrealupperband = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outrealmiddleband = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outreallowerband = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_BBANDS( startidx , endidx , <double *>real.data , timeperiod , nbdevup , nbdevdn , matype , &outbegidx , &outnbelement , <double *>outrealupperband.data , <double *>outrealmiddleband.data , <double *>outreallowerband.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outrealupperband , outrealmiddleband , outreallowerband )

def BETA( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 , timeperiod=-2**31 ):
    """BETA(real0, real1[, timeperiod=?])

    Beta"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_BETA( startidx , endidx , <double *>real0.data , <double *>real1.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def BOP( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """BOP(open, high, low, close)

    Balance Of Power"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_BOP( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def CCI( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """CCI(high, low, close[, timeperiod=?])

    Commodity Channel Index"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_CCI( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def CDL2CROWS( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDL2CROWS(open, high, low, close)

    Two Crows"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDL2CROWS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDL3BLACKCROWS( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDL3BLACKCROWS(open, high, low, close)

    Three Black Crows"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDL3BLACKCROWS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDL3INSIDE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDL3INSIDE(open, high, low, close)

    Three Inside Up/Down"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDL3INSIDE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDL3LINESTRIKE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDL3LINESTRIKE(open, high, low, close)

    Three-Line Strike """
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDL3LINESTRIKE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDL3OUTSIDE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDL3OUTSIDE(open, high, low, close)

    Three Outside Up/Down"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDL3OUTSIDE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDL3STARSINSOUTH( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDL3STARSINSOUTH(open, high, low, close)

    Three Stars In The South"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDL3STARSINSOUTH( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDL3WHITESOLDIERS( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDL3WHITESOLDIERS(open, high, low, close)

    Three Advancing White Soldiers"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDL3WHITESOLDIERS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLABANDONEDBABY( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    """CDLABANDONEDBABY(open, high, low, close[, penetration=?])

    Abandoned Baby"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLABANDONEDBABY( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLADVANCEBLOCK( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLADVANCEBLOCK(open, high, low, close)

    Advance Block"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLADVANCEBLOCK( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLBELTHOLD( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLBELTHOLD(open, high, low, close)

    Belt-hold"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLBELTHOLD( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLBREAKAWAY( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLBREAKAWAY(open, high, low, close)

    Breakaway"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLBREAKAWAY( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLCLOSINGMARUBOZU( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLCLOSINGMARUBOZU(open, high, low, close)

    Closing Marubozu"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLCLOSINGMARUBOZU( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLCONCEALBABYSWALL( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLCONCEALBABYSWALL(open, high, low, close)

    Concealing Baby Swallow"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLCONCEALBABYSWALL( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLCOUNTERATTACK( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLCOUNTERATTACK(open, high, low, close)

    Counterattack"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLCOUNTERATTACK( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLDARKCLOUDCOVER( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    """CDLDARKCLOUDCOVER(open, high, low, close[, penetration=?])

    Dark Cloud Cover"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLDARKCLOUDCOVER( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLDOJI( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLDOJI(open, high, low, close)

    Doji"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLDOJI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLDOJISTAR( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLDOJISTAR(open, high, low, close)

    Doji Star"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLDOJISTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLDRAGONFLYDOJI( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLDRAGONFLYDOJI(open, high, low, close)

    Dragonfly Doji"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLDRAGONFLYDOJI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLENGULFING( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLENGULFING(open, high, low, close)

    Engulfing Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLENGULFING( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLEVENINGDOJISTAR( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    """CDLEVENINGDOJISTAR(open, high, low, close[, penetration=?])

    Evening Doji Star"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLEVENINGDOJISTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLEVENINGSTAR( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    """CDLEVENINGSTAR(open, high, low, close[, penetration=?])

    Evening Star"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLEVENINGSTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLGAPSIDESIDEWHITE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLGAPSIDESIDEWHITE(open, high, low, close)

    Up/Down-gap side-by-side white lines"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLGAPSIDESIDEWHITE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLGRAVESTONEDOJI( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLGRAVESTONEDOJI(open, high, low, close)

    Gravestone Doji"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLGRAVESTONEDOJI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLHAMMER( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLHAMMER(open, high, low, close)

    Hammer"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLHAMMER( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLHANGINGMAN( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLHANGINGMAN(open, high, low, close)

    Hanging Man"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLHANGINGMAN( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLHARAMI( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLHARAMI(open, high, low, close)

    Harami Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLHARAMI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLHARAMICROSS( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLHARAMICROSS(open, high, low, close)

    Harami Cross Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLHARAMICROSS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLHIGHWAVE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLHIGHWAVE(open, high, low, close)

    High-Wave Candle"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLHIGHWAVE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLHIKKAKE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLHIKKAKE(open, high, low, close)

    Hikkake Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLHIKKAKE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLHIKKAKEMOD( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLHIKKAKEMOD(open, high, low, close)

    Modified Hikkake Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLHIKKAKEMOD( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLHOMINGPIGEON( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLHOMINGPIGEON(open, high, low, close)

    Homing Pigeon"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLHOMINGPIGEON( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLIDENTICAL3CROWS( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLIDENTICAL3CROWS(open, high, low, close)

    Identical Three Crows"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLIDENTICAL3CROWS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLINNECK( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLINNECK(open, high, low, close)

    In-Neck Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLINNECK( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLINVERTEDHAMMER( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLINVERTEDHAMMER(open, high, low, close)

    Inverted Hammer"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLINVERTEDHAMMER( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLKICKING( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLKICKING(open, high, low, close)

    Kicking"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLKICKING( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLKICKINGBYLENGTH( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLKICKINGBYLENGTH(open, high, low, close)

    Kicking - bull/bear determined by the longer marubozu"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLKICKINGBYLENGTH( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLLADDERBOTTOM( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLLADDERBOTTOM(open, high, low, close)

    Ladder Bottom"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLLADDERBOTTOM( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLLONGLEGGEDDOJI( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLLONGLEGGEDDOJI(open, high, low, close)

    Long Legged Doji"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLLONGLEGGEDDOJI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLLONGLINE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLLONGLINE(open, high, low, close)

    Long Line Candle"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLLONGLINE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLMARUBOZU( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLMARUBOZU(open, high, low, close)

    Marubozu"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLMARUBOZU( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLMATCHINGLOW( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLMATCHINGLOW(open, high, low, close)

    Matching Low"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLMATCHINGLOW( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLMATHOLD( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    """CDLMATHOLD(open, high, low, close[, penetration=?])

    Mat Hold"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLMATHOLD( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLMORNINGDOJISTAR( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    """CDLMORNINGDOJISTAR(open, high, low, close[, penetration=?])

    Morning Doji Star"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLMORNINGDOJISTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLMORNINGSTAR( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , penetration=-4e37 ):
    """CDLMORNINGSTAR(open, high, low, close[, penetration=?])

    Morning Star"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLMORNINGSTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , penetration , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLONNECK( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLONNECK(open, high, low, close)

    On-Neck Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLONNECK( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLPIERCING( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLPIERCING(open, high, low, close)

    Piercing Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLPIERCING( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLRICKSHAWMAN( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLRICKSHAWMAN(open, high, low, close)

    Rickshaw Man"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLRICKSHAWMAN( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLRISEFALL3METHODS( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLRISEFALL3METHODS(open, high, low, close)

    Rising/Falling Three Methods"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLRISEFALL3METHODS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLSEPARATINGLINES( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLSEPARATINGLINES(open, high, low, close)

    Separating Lines"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLSEPARATINGLINES( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLSHOOTINGSTAR( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLSHOOTINGSTAR(open, high, low, close)

    Shooting Star"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLSHOOTINGSTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLSHORTLINE( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLSHORTLINE(open, high, low, close)

    Short Line Candle"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLSHORTLINE( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLSPINNINGTOP( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLSPINNINGTOP(open, high, low, close)

    Spinning Top"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLSPINNINGTOP( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLSTALLEDPATTERN( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLSTALLEDPATTERN(open, high, low, close)

    Stalled Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLSTALLEDPATTERN( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLSTICKSANDWICH( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLSTICKSANDWICH(open, high, low, close)

    Stick Sandwich"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLSTICKSANDWICH( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLTAKURI( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLTAKURI(open, high, low, close)

    Takuri (Dragonfly Doji with very long lower shadow)"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLTAKURI( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLTASUKIGAP( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLTASUKIGAP(open, high, low, close)

    Tasuki Gap"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLTASUKIGAP( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLTHRUSTING( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLTHRUSTING(open, high, low, close)

    Thrusting Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLTHRUSTING( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLTRISTAR( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLTRISTAR(open, high, low, close)

    Tristar Pattern"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLTRISTAR( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLUNIQUE3RIVER( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLUNIQUE3RIVER(open, high, low, close)

    Unique 3 River"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLUNIQUE3RIVER( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLUPSIDEGAP2CROWS( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLUPSIDEGAP2CROWS(open, high, low, close)

    Upside Gap Two Crows"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLUPSIDEGAP2CROWS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CDLXSIDEGAP3METHODS( np.ndarray[np.float_t, ndim=1] open , np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """CDLXSIDEGAP3METHODS(open, high, low, close)

    Upside/Downside Gap Three Methods"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_CDLXSIDEGAP3METHODS( startidx , endidx , <double *>open.data , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def CEIL( np.ndarray[np.float_t, ndim=1] real ):
    """CEIL(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_CEIL( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def CMO( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """CMO(real[, timeperiod=?])

    Chande Momentum Oscillator"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_CMO( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def CORREL( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 , timeperiod=-2**31 ):
    """CORREL(real0, real1[, timeperiod=?])

    Pearson's Correlation Coefficient (r)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_CORREL( startidx , endidx , <double *>real0.data , <double *>real1.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def COS( np.ndarray[np.float_t, ndim=1] real ):
    """COS(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_COS( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def COSH( np.ndarray[np.float_t, ndim=1] real ):
    """COSH(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_COSH( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def DEMA( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """DEMA(real[, timeperiod=?])

    Double Exponential Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_DEMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def DIV( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 ):
    """DIV(real0, real1)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_DIV( startidx , endidx , <double *>real0.data , <double *>real1.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def DX( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """DX(high, low, close[, timeperiod=?])

    Directional Movement Index"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_DX( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def EMA( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """EMA(real[, timeperiod=?])

    Exponential Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_EMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def EXP( np.ndarray[np.float_t, ndim=1] real ):
    """EXP(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_EXP( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def FLOOR( np.ndarray[np.float_t, ndim=1] real ):
    """FLOOR(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_FLOOR( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def HT_DCPERIOD( np.ndarray[np.float_t, ndim=1] real ):
    """HT_DCPERIOD(real)

    Hilbert Transform - Dominant Cycle Period"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_HT_DCPERIOD( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def HT_DCPHASE( np.ndarray[np.float_t, ndim=1] real ):
    """HT_DCPHASE(real)

    Hilbert Transform - Dominant Cycle Phase"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_HT_DCPHASE( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def HT_PHASOR( np.ndarray[np.float_t, ndim=1] real ):
    """HT_PHASOR(real)

    Hilbert Transform - Phasor Components"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outinphase = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outquadrature = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_HT_PHASOR( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outinphase.data , <double *>outquadrature.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinphase , outquadrature )

def HT_SINE( np.ndarray[np.float_t, ndim=1] real ):
    """HT_SINE(real)

    Hilbert Transform - SineWave"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outsine = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outleadsine = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_HT_SINE( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outsine.data , <double *>outleadsine.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outsine , outleadsine )

def HT_TRENDLINE( np.ndarray[np.float_t, ndim=1] real ):
    """HT_TRENDLINE(real)

    Hilbert Transform - Instantaneous Trendline"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_HT_TRENDLINE( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def HT_TRENDMODE( np.ndarray[np.float_t, ndim=1] real ):
    """HT_TRENDMODE(real)

    Hilbert Transform - Trend vs Cycle Mode"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_HT_TRENDMODE( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def KAMA( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """KAMA(real[, timeperiod=?])

    Kaufman Adaptive Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_KAMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def LINEARREG( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """LINEARREG(real[, timeperiod=?])

    Linear Regression"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_LINEARREG( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def LINEARREG_ANGLE( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """LINEARREG_ANGLE(real[, timeperiod=?])

    Linear Regression Angle"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_LINEARREG_ANGLE( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def LINEARREG_INTERCEPT( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """LINEARREG_INTERCEPT(real[, timeperiod=?])

    Linear Regression Intercept"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_LINEARREG_INTERCEPT( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def LINEARREG_SLOPE( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """LINEARREG_SLOPE(real[, timeperiod=?])

    Linear Regression Slope"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_LINEARREG_SLOPE( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def LN( np.ndarray[np.float_t, ndim=1] real ):
    """LN(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_LN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def LOG10( np.ndarray[np.float_t, ndim=1] real ):
    """LOG10(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_LOG10( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MA( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , matype=0 ):
    """MA(real[, timeperiod=?, matype=?])

    All Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MA( startidx , endidx , <double *>real.data , timeperiod , matype , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MACD( np.ndarray[np.float_t, ndim=1] real , fastperiod=-2**31 , slowperiod=-2**31 , signalperiod=-2**31 ):
    """MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])

    Moving Average Convergence/Divergence"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outmacd = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outmacdsignal = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outmacdhist = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MACD( startidx , endidx , <double *>real.data , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , <double *>outmacd.data , <double *>outmacdsignal.data , <double *>outmacdhist.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outmacd , outmacdsignal , outmacdhist )

def MACDEXT( np.ndarray[np.float_t, ndim=1] real , fastperiod=-2**31 , fastmatype=0 , slowperiod=-2**31 , slowmatype=0 , signalperiod=-2**31 , signalmatype=0 ):
    """MACDEXT(real[, fastperiod=?, fastmatype=?, slowperiod=?, slowmatype=?, signalperiod=?, signalmatype=?])

    MACD with controllable MA type"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outmacd = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outmacdsignal = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outmacdhist = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MACDEXT( startidx , endidx , <double *>real.data , fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype , &outbegidx , &outnbelement , <double *>outmacd.data , <double *>outmacdsignal.data , <double *>outmacdhist.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outmacd , outmacdsignal , outmacdhist )

def MACDFIX( np.ndarray[np.float_t, ndim=1] real , signalperiod=-2**31 ):
    """MACDFIX(real[, signalperiod=?])

    Moving Average Convergence/Divergence Fix 12/26"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outmacd = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outmacdsignal = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outmacdhist = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MACDFIX( startidx , endidx , <double *>real.data , signalperiod , &outbegidx , &outnbelement , <double *>outmacd.data , <double *>outmacdsignal.data , <double *>outmacdhist.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outmacd , outmacdsignal , outmacdhist )

def MAMA( np.ndarray[np.float_t, ndim=1] real , fastlimit=-4e37 , slowlimit=-4e37 ):
    """MAMA(real[, fastlimit=?, slowlimit=?])

    MESA Adaptive Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outmama = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outfama = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MAMA( startidx , endidx , <double *>real.data , fastlimit , slowlimit , &outbegidx , &outnbelement , <double *>outmama.data , <double *>outfama.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outmama , outfama )

def MAVP( np.ndarray[np.float_t, ndim=1] real , np.ndarray[np.float_t, ndim=1] periods , minperiod=-2**31 , maxperiod=-2**31 , matype=0 ):
    """MAVP(real, periods[, minperiod=?, maxperiod=?, matype=?])"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MAVP( startidx , endidx , <double *>real.data , <double *>periods.data , minperiod , maxperiod , matype , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MAX( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """MAX(real[, timeperiod=?])

    Highest value over a specified period"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MAX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MAXINDEX( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """MAXINDEX(real[, timeperiod=?])

    Index of highest value over a specified period"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_MAXINDEX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def MEDPRICE( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low ):
    """MEDPRICE(high, low)

    Median Price"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MEDPRICE( startidx , endidx , <double *>high.data , <double *>low.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MFI( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , np.ndarray[np.float_t, ndim=1] volume , timeperiod=-2**31 ):
    """MFI(high, low, close, volume[, timeperiod=?])

    Money Flow Index"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MFI( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , <double *>volume.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MIDPOINT( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """MIDPOINT(real[, timeperiod=?])

    MidPoint over period"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MIDPOINT( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MIDPRICE( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    """MIDPRICE(high, low[, timeperiod=?])

    Midpoint Price over period"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MIDPRICE( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MIN( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """MIN(real[, timeperiod=?])

    Lowest value over a specified period"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MIN( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MININDEX( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """MININDEX(real[, timeperiod=?])

    Index of lowest value over a specified period"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outinteger = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_MININDEX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <int *>outinteger.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outinteger )

def MINMAX( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """MINMAX(real[, timeperiod=?])

    Lowest and highest values over a specified period"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outmin = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outmax = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MINMAX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outmin.data , <double *>outmax.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outmin , outmax )

def MINMAXINDEX( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """MINMAXINDEX(real[, timeperiod=?])

    Indexes of lowest and highest values over a specified period"""
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
    cdef np.ndarray[np.int32_t, ndim=1] outminidx = numpy.zeros(allocation, dtype="i4")
    cdef np.ndarray[np.int32_t, ndim=1] outmaxidx = numpy.zeros(allocation, dtype="i4")
    TA_Initialize()
    retCode = TA_MINMAXINDEX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <int *>outminidx.data , <int *>outmaxidx.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outminidx , outmaxidx )

def MINUS_DI( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """MINUS_DI(high, low, close[, timeperiod=?])

    Minus Directional Indicator"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MINUS_DI( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MINUS_DM( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    """MINUS_DM(high, low[, timeperiod=?])

    Minus Directional Movement"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MINUS_DM( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MOM( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """MOM(real[, timeperiod=?])

    Momentum"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MOM( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def MULT( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 ):
    """MULT(real0, real1)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_MULT( startidx , endidx , <double *>real0.data , <double *>real1.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def NATR( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """NATR(high, low, close[, timeperiod=?])

    Normalized Average True Range"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_NATR( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def OBV( np.ndarray[np.float_t, ndim=1] real , np.ndarray[np.float_t, ndim=1] volume ):
    """OBV(real, volume)

    On Balance Volume"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_OBV( startidx , endidx , <double *>real.data , <double *>volume.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def PLUS_DI( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """PLUS_DI(high, low, close[, timeperiod=?])

    Plus Directional Indicator"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_PLUS_DI( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def PLUS_DM( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , timeperiod=-2**31 ):
    """PLUS_DM(high, low[, timeperiod=?])

    Plus Directional Movement"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_PLUS_DM( startidx , endidx , <double *>high.data , <double *>low.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def PPO( np.ndarray[np.float_t, ndim=1] real , fastperiod=-2**31 , slowperiod=-2**31 , matype=0 ):
    """PPO(real[, fastperiod=?, slowperiod=?, matype=?])

    Percentage Price Oscillator"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_PPO( startidx , endidx , <double *>real.data , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ROC( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """ROC(real[, timeperiod=?])

    Rate of change : ((price/prevPrice)-1)*100"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ROC( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ROCP( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """ROCP(real[, timeperiod=?])

    Rate of change Percentage: (price-prevPrice)/prevPrice"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ROCP( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ROCR( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """ROCR(real[, timeperiod=?])

    Rate of change ratio: (price/prevPrice)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ROCR( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ROCR100( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """ROCR100(real[, timeperiod=?])

    Rate of change ratio 100 scale: (price/prevPrice)*100"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ROCR100( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def RSI( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """RSI(real[, timeperiod=?])

    Relative Strength Index"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_RSI( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def SAR( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , acceleration=-4e37 , maximum=-4e37 ):
    """SAR(high, low[, acceleration=?, maximum=?])

    Parabolic SAR"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_SAR( startidx , endidx , <double *>high.data , <double *>low.data , acceleration , maximum , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def SAREXT( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , startvalue=-4e37 , offsetonreverse=-4e37 , accelerationinitlong=-4e37 , accelerationlong=-4e37 , accelerationmaxlong=-4e37 , accelerationinitshort=-4e37 , accelerationshort=-4e37 , accelerationmaxshort=-4e37 ):
    """SAREXT(high, low[, startvalue=?, offsetonreverse=?, accelerationinitlong=?, accelerationlong=?, accelerationmaxlong=?, accelerationinitshort=?, accelerationshort=?, accelerationmaxshort=?])

    Parabolic SAR - Extended"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_SAREXT( startidx , endidx , <double *>high.data , <double *>low.data , startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def SIN( np.ndarray[np.float_t, ndim=1] real ):
    """SIN(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_SIN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def SINH( np.ndarray[np.float_t, ndim=1] real ):
    """SINH(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_SINH( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def SMA( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """SMA(real[, timeperiod=?])

    Simple Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_SMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def SQRT( np.ndarray[np.float_t, ndim=1] real ):
    """SQRT(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_SQRT( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def STDDEV( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , nbdev=-4e37 ):
    """STDDEV(real[, timeperiod=?, nbdev=?])

    Standard Deviation"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_STDDEV( startidx , endidx , <double *>real.data , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def STOCH( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , fastk_period=-2**31 , slowk_period=-2**31 , slowk_matype=0 , slowd_period=-2**31 , slowd_matype=0 ):
    """STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])

    Stochastic"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outslowk = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outslowd = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_STOCH( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , <double *>outslowk.data , <double *>outslowd.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outslowk , outslowd )

def STOCHF( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , fastk_period=-2**31 , fastd_period=-2**31 , fastd_matype=0 ):
    """STOCHF(high, low, close[, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Fast"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outfastk = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outfastd = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_STOCHF( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>outfastk.data , <double *>outfastd.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outfastk , outfastd )

def STOCHRSI( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , fastk_period=-2**31 , fastd_period=-2**31 , fastd_matype=0 ):
    """STOCHRSI(real[, timeperiod=?, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Relative Strength Index"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outfastk = numpy.zeros(allocation, dtype="f8")
    cdef np.ndarray[np.float64_t, ndim=1] outfastd = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_STOCHRSI( startidx , endidx , <double *>real.data , timeperiod , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>outfastk.data , <double *>outfastd.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outfastk , outfastd )

def SUB( np.ndarray[np.float_t, ndim=1] real0 , np.ndarray[np.float_t, ndim=1] real1 ):
    """SUB(real0, real1)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_SUB( startidx , endidx , <double *>real0.data , <double *>real1.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def SUM( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """SUM(real[, timeperiod=?])

    Summation"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_SUM( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def T3( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , vfactor=-4e37 ):
    """T3(real[, timeperiod=?, vfactor=?])

    Triple Exponential Moving Average (T3)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_T3( startidx , endidx , <double *>real.data , timeperiod , vfactor , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def TAN( np.ndarray[np.float_t, ndim=1] real ):
    """TAN(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_TAN( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def TANH( np.ndarray[np.float_t, ndim=1] real ):
    """TANH(real)"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_TANH( startidx , endidx , <double *>real.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def TEMA( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """TEMA(real[, timeperiod=?])

    Triple Exponential Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_TEMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def TRANGE( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """TRANGE(high, low, close)

    True Range"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_TRANGE( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def TRIMA( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """TRIMA(real[, timeperiod=?])

    Triangular Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_TRIMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def TRIX( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """TRIX(real[, timeperiod=?])

    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_TRIX( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def TSF( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """TSF(real[, timeperiod=?])

    Time Series Forecast"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_TSF( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def TYPPRICE( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """TYPPRICE(high, low, close)

    Typical Price"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_TYPPRICE( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def ULTOSC( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod1=-2**31 , timeperiod2=-2**31 , timeperiod3=-2**31 ):
    """ULTOSC(high, low, close[, timeperiod1=?, timeperiod2=?, timeperiod3=?])

    Ultimate Oscillator"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_ULTOSC( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod1 , timeperiod2 , timeperiod3 , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def VAR( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 , nbdev=-4e37 ):
    """VAR(real[, timeperiod=?, nbdev=?])

    Variance"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_VAR( startidx , endidx , <double *>real.data , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def WCLPRICE( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close ):
    """WCLPRICE(high, low, close)

    Weighted Close Price"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_WCLPRICE( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def WILLR( np.ndarray[np.float_t, ndim=1] high , np.ndarray[np.float_t, ndim=1] low , np.ndarray[np.float_t, ndim=1] close , timeperiod=-2**31 ):
    """WILLR(high, low, close[, timeperiod=?])

    Williams' %R"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_WILLR( startidx , endidx , <double *>high.data , <double *>low.data , <double *>close.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

def WMA( np.ndarray[np.float_t, ndim=1] real , timeperiod=-2**31 ):
    """WMA(real[, timeperiod=?])

    Weighted Moving Average"""
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
    cdef np.ndarray[np.float64_t, ndim=1] outreal = numpy.zeros(allocation, dtype="f8")
    TA_Initialize()
    retCode = TA_WMA( startidx , endidx , <double *>real.data , timeperiod , &outbegidx , &outnbelement , <double *>outreal.data )
    if retCode != TA_SUCCESS:
        raise Exception("%d" % retCode)
    TA_Shutdown()
    return ( outbegidx , outreal )

__all__ = ["ACOS","AD","ADD","ADOSC","ADX","ADXR","APO","AROON","AROONOSC","ASIN","ATAN","ATR","AVGPRICE","BBANDS","BETA","BOP","CCI","CDL2CROWS","CDL3BLACKCROWS","CDL3INSIDE","CDL3LINESTRIKE","CDL3OUTSIDE","CDL3STARSINSOUTH","CDL3WHITESOLDIERS","CDLABANDONEDBABY","CDLADVANCEBLOCK","CDLBELTHOLD","CDLBREAKAWAY","CDLCLOSINGMARUBOZU","CDLCONCEALBABYSWALL","CDLCOUNTERATTACK","CDLDARKCLOUDCOVER","CDLDOJI","CDLDOJISTAR","CDLDRAGONFLYDOJI","CDLENGULFING","CDLEVENINGDOJISTAR","CDLEVENINGSTAR","CDLGAPSIDESIDEWHITE","CDLGRAVESTONEDOJI","CDLHAMMER","CDLHANGINGMAN","CDLHARAMI","CDLHARAMICROSS","CDLHIGHWAVE","CDLHIKKAKE","CDLHIKKAKEMOD","CDLHOMINGPIGEON","CDLIDENTICAL3CROWS","CDLINNECK","CDLINVERTEDHAMMER","CDLKICKING","CDLKICKINGBYLENGTH","CDLLADDERBOTTOM","CDLLONGLEGGEDDOJI","CDLLONGLINE","CDLMARUBOZU","CDLMATCHINGLOW","CDLMATHOLD","CDLMORNINGDOJISTAR","CDLMORNINGSTAR","CDLONNECK","CDLPIERCING","CDLRICKSHAWMAN","CDLRISEFALL3METHODS","CDLSEPARATINGLINES","CDLSHOOTINGSTAR","CDLSHORTLINE","CDLSPINNINGTOP","CDLSTALLEDPATTERN","CDLSTICKSANDWICH","CDLTAKURI","CDLTASUKIGAP","CDLTHRUSTING","CDLTRISTAR","CDLUNIQUE3RIVER","CDLUPSIDEGAP2CROWS","CDLXSIDEGAP3METHODS","CEIL","CMO","CORREL","COS","COSH","DEMA","DIV","DX","EMA","EXP","FLOOR","HT_DCPERIOD","HT_DCPHASE","HT_PHASOR","HT_SINE","HT_TRENDLINE","HT_TRENDMODE","KAMA","LINEARREG","LINEARREG_ANGLE","LINEARREG_INTERCEPT","LINEARREG_SLOPE","LN","LOG10","MA","MACD","MACDEXT","MACDFIX","MAMA","MAVP","MAX","MAXINDEX","MEDPRICE","MFI","MIDPOINT","MIDPRICE","MIN","MININDEX","MINMAX","MINMAXINDEX","MINUS_DI","MINUS_DM","MOM","MULT","NATR","OBV","PLUS_DI","PLUS_DM","PPO","ROC","ROCP","ROCR","ROCR100","RSI","SAR","SAREXT","SIN","SINH","SMA","SQRT","STDDEV","STOCH","STOCHF","STOCHRSI","SUB","SUM","T3","TAN","TANH","TEMA","TRANGE","TRIMA","TRIX","TSF","TYPPRICE","ULTOSC","VAR","WCLPRICE","WILLR","WMA"]
