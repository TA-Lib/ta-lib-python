
import numpy
cimport numpy as np

ctypedef int TA_RetCode
ctypedef int TA_MAType

# extract the needed part of ta_libc.h that I will use in the interface
cdef extern from "ta_libc.h":
    enum: TA_SUCCESS
    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()
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

def ACOS( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_ACOS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ACOS( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def AD( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , np.ndarray[np.float_t, ndim=1] inVolume ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_AD_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_AD( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , <double *>inVolume.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ADD( np.ndarray[np.float_t, ndim=1] inReal0 , np.ndarray[np.float_t, ndim=1] inReal1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal0.shape[0] - 1
    cdef int lookback = TA_ADD_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ADD( startIdx , endIdx , <double *>inReal0.data , <double *>inReal1.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ADOSC( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , np.ndarray[np.float_t, ndim=1] inVolume , optInFastPeriod=1 , optInSlowPeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_ADOSC_Lookback( optInFastPeriod , optInSlowPeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ADOSC( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , <double *>inVolume.data , optInFastPeriod , optInSlowPeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ADX( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_ADX_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ADX( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ADXR( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_ADXR_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ADXR( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def APO( np.ndarray[np.float_t, ndim=1] inReal , optInFastPeriod=1 , optInSlowPeriod=1 , optInMAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_APO_Lookback( optInFastPeriod , optInSlowPeriod , optInMAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_APO( startIdx , endIdx , <double *>inReal.data , optInFastPeriod , optInSlowPeriod , optInMAType , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def AROON( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_AROON_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outAroonDown = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outAroonUp = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_AROON( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outAroonDown.data , <double *>outAroonUp.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outAroonDown , outAroonUp )

def AROONOSC( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_AROONOSC_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_AROONOSC( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ASIN( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_ASIN_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ASIN( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ATAN( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_ATAN_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ATAN( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ATR( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_ATR_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ATR( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def AVGPRICE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_AVGPRICE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_AVGPRICE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def BBANDS( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 , optInNbDevUp=1 , optInNbDevDn=1 , optInMAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_BBANDS_Lookback( optInTimePeriod , optInNbDevUp , optInNbDevDn , optInMAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outRealUpperBand = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outRealMiddleBand = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outRealLowerBand = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_BBANDS( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , optInNbDevUp , optInNbDevDn , optInMAType , &outBegIdx , &outNBElement , <double *>outRealUpperBand.data , <double *>outRealMiddleBand.data , <double *>outRealLowerBand.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outRealUpperBand , outRealMiddleBand , outRealLowerBand )

def BETA( np.ndarray[np.float_t, ndim=1] inReal0 , np.ndarray[np.float_t, ndim=1] inReal1 , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal0.shape[0] - 1
    cdef int lookback = TA_BETA_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_BETA( startIdx , endIdx , <double *>inReal0.data , <double *>inReal1.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def BOP( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_BOP_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_BOP( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def CCI( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CCI_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CCI( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def CDL2CROWS( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDL2CROWS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDL2CROWS( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDL3BLACKCROWS( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDL3BLACKCROWS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDL3BLACKCROWS( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDL3INSIDE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDL3INSIDE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDL3INSIDE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDL3LINESTRIKE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDL3LINESTRIKE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDL3LINESTRIKE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDL3OUTSIDE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDL3OUTSIDE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDL3OUTSIDE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDL3STARSINSOUTH( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDL3STARSINSOUTH_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDL3STARSINSOUTH( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDL3WHITESOLDIERS( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDL3WHITESOLDIERS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDL3WHITESOLDIERS( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLABANDONEDBABY( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInPenetration ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLABANDONEDBABY_Lookback( optInPenetration )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLABANDONEDBABY( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInPenetration , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLADVANCEBLOCK( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLADVANCEBLOCK_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLADVANCEBLOCK( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLBELTHOLD( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLBELTHOLD_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLBELTHOLD( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLBREAKAWAY( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLBREAKAWAY_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLBREAKAWAY( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLCLOSINGMARUBOZU( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLCLOSINGMARUBOZU_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLCLOSINGMARUBOZU( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLCONCEALBABYSWALL( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLCONCEALBABYSWALL_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLCONCEALBABYSWALL( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLCOUNTERATTACK( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLCOUNTERATTACK_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLCOUNTERATTACK( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLDARKCLOUDCOVER( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInPenetration ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLDARKCLOUDCOVER_Lookback( optInPenetration )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLDARKCLOUDCOVER( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInPenetration , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLDOJI( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLDOJI_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLDOJI( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLDOJISTAR( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLDOJISTAR_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLDOJISTAR( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLDRAGONFLYDOJI( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLDRAGONFLYDOJI_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLDRAGONFLYDOJI( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLENGULFING( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLENGULFING_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLENGULFING( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLEVENINGDOJISTAR( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInPenetration ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLEVENINGDOJISTAR_Lookback( optInPenetration )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLEVENINGDOJISTAR( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInPenetration , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLEVENINGSTAR( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInPenetration ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLEVENINGSTAR_Lookback( optInPenetration )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLEVENINGSTAR( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInPenetration , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLGAPSIDESIDEWHITE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLGAPSIDESIDEWHITE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLGAPSIDESIDEWHITE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLGRAVESTONEDOJI( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLGRAVESTONEDOJI_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLGRAVESTONEDOJI( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLHAMMER( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLHAMMER_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLHAMMER( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLHANGINGMAN( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLHANGINGMAN_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLHANGINGMAN( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLHARAMI( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLHARAMI_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLHARAMI( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLHARAMICROSS( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLHARAMICROSS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLHARAMICROSS( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLHIGHWAVE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLHIGHWAVE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLHIGHWAVE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLHIKKAKE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLHIKKAKE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLHIKKAKE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLHIKKAKEMOD( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLHIKKAKEMOD_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLHIKKAKEMOD( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLHOMINGPIGEON( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLHOMINGPIGEON_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLHOMINGPIGEON( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLIDENTICAL3CROWS( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLIDENTICAL3CROWS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLIDENTICAL3CROWS( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLINNECK( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLINNECK_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLINNECK( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLINVERTEDHAMMER( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLINVERTEDHAMMER_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLINVERTEDHAMMER( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLKICKING( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLKICKING_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLKICKING( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLKICKINGBYLENGTH( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLKICKINGBYLENGTH_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLKICKINGBYLENGTH( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLLADDERBOTTOM( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLLADDERBOTTOM_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLLADDERBOTTOM( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLLONGLEGGEDDOJI( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLLONGLEGGEDDOJI_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLLONGLEGGEDDOJI( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLLONGLINE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLLONGLINE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLLONGLINE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLMARUBOZU( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLMARUBOZU_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLMARUBOZU( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLMATCHINGLOW( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLMATCHINGLOW_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLMATCHINGLOW( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLMATHOLD( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInPenetration ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLMATHOLD_Lookback( optInPenetration )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLMATHOLD( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInPenetration , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLMORNINGDOJISTAR( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInPenetration ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLMORNINGDOJISTAR_Lookback( optInPenetration )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLMORNINGDOJISTAR( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInPenetration , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLMORNINGSTAR( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInPenetration ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLMORNINGSTAR_Lookback( optInPenetration )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLMORNINGSTAR( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInPenetration , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLONNECK( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLONNECK_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLONNECK( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLPIERCING( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLPIERCING_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLPIERCING( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLRICKSHAWMAN( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLRICKSHAWMAN_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLRICKSHAWMAN( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLRISEFALL3METHODS( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLRISEFALL3METHODS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLRISEFALL3METHODS( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLSEPARATINGLINES( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLSEPARATINGLINES_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLSEPARATINGLINES( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLSHOOTINGSTAR( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLSHOOTINGSTAR_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLSHOOTINGSTAR( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLSHORTLINE( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLSHORTLINE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLSHORTLINE( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLSPINNINGTOP( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLSPINNINGTOP_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLSPINNINGTOP( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLSTALLEDPATTERN( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLSTALLEDPATTERN_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLSTALLEDPATTERN( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLSTICKSANDWICH( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLSTICKSANDWICH_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLSTICKSANDWICH( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLTAKURI( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLTAKURI_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLTAKURI( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLTASUKIGAP( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLTASUKIGAP_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLTASUKIGAP( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLTHRUSTING( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLTHRUSTING_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLTHRUSTING( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLTRISTAR( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLTRISTAR_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLTRISTAR( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLUNIQUE3RIVER( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLUNIQUE3RIVER_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLUNIQUE3RIVER( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLUPSIDEGAP2CROWS( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLUPSIDEGAP2CROWS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLUPSIDEGAP2CROWS( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CDLXSIDEGAP3METHODS( np.ndarray[np.float_t, ndim=1] inOpen , np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_CDLXSIDEGAP3METHODS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CDLXSIDEGAP3METHODS( startIdx , endIdx , <double *>inOpen.data , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def CEIL( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_CEIL_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CEIL( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def CMO( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_CMO_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CMO( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def CORREL( np.ndarray[np.float_t, ndim=1] inReal0 , np.ndarray[np.float_t, ndim=1] inReal1 , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal0.shape[0] - 1
    cdef int lookback = TA_CORREL_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_CORREL( startIdx , endIdx , <double *>inReal0.data , <double *>inReal1.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def COS( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_COS_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_COS( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def COSH( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_COSH_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_COSH( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def DEMA( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_DEMA_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_DEMA( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def DIV( np.ndarray[np.float_t, ndim=1] inReal0 , np.ndarray[np.float_t, ndim=1] inReal1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal0.shape[0] - 1
    cdef int lookback = TA_DIV_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_DIV( startIdx , endIdx , <double *>inReal0.data , <double *>inReal1.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def DX( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_DX_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_DX( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def EMA( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_EMA_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_EMA( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def EXP( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_EXP_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_EXP( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def FLOOR( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_FLOOR_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_FLOOR( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def HT_DCPERIOD( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_HT_DCPERIOD_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_HT_DCPERIOD( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def HT_DCPHASE( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_HT_DCPHASE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_HT_DCPHASE( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def HT_PHASOR( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_HT_PHASOR_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outInPhase = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outQuadrature = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_HT_PHASOR( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outInPhase.data , <double *>outQuadrature.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInPhase , outQuadrature )

def HT_SINE( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_HT_SINE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outSine = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outLeadSine = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_HT_SINE( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outSine.data , <double *>outLeadSine.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outSine , outLeadSine )

def HT_TRENDLINE( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_HT_TRENDLINE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_HT_TRENDLINE( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def HT_TRENDMODE( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_HT_TRENDMODE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_HT_TRENDMODE( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def KAMA( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_KAMA_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_KAMA( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def LINEARREG( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_LINEARREG_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_LINEARREG( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def LINEARREG_ANGLE( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_LINEARREG_ANGLE_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_LINEARREG_ANGLE( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def LINEARREG_INTERCEPT( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_LINEARREG_INTERCEPT_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_LINEARREG_INTERCEPT( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def LINEARREG_SLOPE( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_LINEARREG_SLOPE_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_LINEARREG_SLOPE( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def LN( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_LN_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_LN( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def LOG10( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_LOG10_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_LOG10( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MA( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 , optInMAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MA_Lookback( optInTimePeriod , optInMAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MA( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , optInMAType , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MACD( np.ndarray[np.float_t, ndim=1] inReal , optInFastPeriod=1 , optInSlowPeriod=1 , optInSignalPeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MACD_Lookback( optInFastPeriod , optInSlowPeriod , optInSignalPeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outMACD = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outMACDSignal = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outMACDHist = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MACD( startIdx , endIdx , <double *>inReal.data , optInFastPeriod , optInSlowPeriod , optInSignalPeriod , &outBegIdx , &outNBElement , <double *>outMACD.data , <double *>outMACDSignal.data , <double *>outMACDHist.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outMACD , outMACDSignal , outMACDHist )

def MACDEXT( np.ndarray[np.float_t, ndim=1] inReal , optInFastPeriod=1 , optInFastMAType=1 , optInSlowPeriod=1 , optInSlowMAType=1 , optInSignalPeriod=1 , optInSignalMAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MACDEXT_Lookback( optInFastPeriod , optInFastMAType , optInSlowPeriod , optInSlowMAType , optInSignalPeriod , optInSignalMAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outMACD = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outMACDSignal = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outMACDHist = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MACDEXT( startIdx , endIdx , <double *>inReal.data , optInFastPeriod , optInFastMAType , optInSlowPeriod , optInSlowMAType , optInSignalPeriod , optInSignalMAType , &outBegIdx , &outNBElement , <double *>outMACD.data , <double *>outMACDSignal.data , <double *>outMACDHist.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outMACD , outMACDSignal , outMACDHist )

def MACDFIX( np.ndarray[np.float_t, ndim=1] inReal , optInSignalPeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MACDFIX_Lookback( optInSignalPeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outMACD = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outMACDSignal = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outMACDHist = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MACDFIX( startIdx , endIdx , <double *>inReal.data , optInSignalPeriod , &outBegIdx , &outNBElement , <double *>outMACD.data , <double *>outMACDSignal.data , <double *>outMACDHist.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outMACD , outMACDSignal , outMACDHist )

def MAMA( np.ndarray[np.float_t, ndim=1] inReal , optInFastLimit , optInSlowLimit ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MAMA_Lookback( optInFastLimit , optInSlowLimit )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outMAMA = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outFAMA = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MAMA( startIdx , endIdx , <double *>inReal.data , optInFastLimit , optInSlowLimit , &outBegIdx , &outNBElement , <double *>outMAMA.data , <double *>outFAMA.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outMAMA , outFAMA )

def MAVP( np.ndarray[np.float_t, ndim=1] inReal , np.ndarray[np.float_t, ndim=1] inPeriods , optInMinPeriod , optInMaxPeriod , optInMAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MAVP_Lookback( optInMinPeriod , optInMaxPeriod , optInMAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MAVP( startIdx , endIdx , <double *>inReal.data , <double *>inPeriods.data , optInMinPeriod , optInMaxPeriod , optInMAType , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MAX( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MAX_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MAX( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MAXINDEX( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MAXINDEX_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MAXINDEX( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def MEDPRICE( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_MEDPRICE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MEDPRICE( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MFI( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , np.ndarray[np.float_t, ndim=1] inVolume , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_MFI_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MFI( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , <double *>inVolume.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MIDPOINT( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MIDPOINT_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MIDPOINT( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MIDPRICE( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_MIDPRICE_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MIDPRICE( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MIN( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MIN_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MIN( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MININDEX( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MININDEX_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outInteger = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MININDEX( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <int *>outInteger.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outInteger )

def MINMAX( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MINMAX_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outMin = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outMax = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MINMAX( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outMin.data , <double *>outMax.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outMin , outMax )

def MINMAXINDEX( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MINMAXINDEX_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.int_t, ndim=1] outMinIdx = numpy.zeros(allocationSize)
    cdef np.ndarray[np.int_t, ndim=1] outMaxIdx = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MINMAXINDEX( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <int *>outMinIdx.data , <int *>outMaxIdx.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outMinIdx , outMaxIdx )

def MINUS_DI( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_MINUS_DI_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MINUS_DI( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MINUS_DM( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_MINUS_DM_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MINUS_DM( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MOM( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_MOM_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MOM( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def MULT( np.ndarray[np.float_t, ndim=1] inReal0 , np.ndarray[np.float_t, ndim=1] inReal1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal0.shape[0] - 1
    cdef int lookback = TA_MULT_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_MULT( startIdx , endIdx , <double *>inReal0.data , <double *>inReal1.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def NATR( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_NATR_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_NATR( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def OBV( np.ndarray[np.float_t, ndim=1] inReal , np.ndarray[np.float_t, ndim=1] inVolume ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_OBV_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_OBV( startIdx , endIdx , <double *>inReal.data , <double *>inVolume.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def PLUS_DI( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_PLUS_DI_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_PLUS_DI( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def PLUS_DM( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_PLUS_DM_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_PLUS_DM( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def PPO( np.ndarray[np.float_t, ndim=1] inReal , optInFastPeriod=1 , optInSlowPeriod=1 , optInMAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_PPO_Lookback( optInFastPeriod , optInSlowPeriod , optInMAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_PPO( startIdx , endIdx , <double *>inReal.data , optInFastPeriod , optInSlowPeriod , optInMAType , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ROC( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_ROC_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ROC( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ROCP( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_ROCP_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ROCP( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ROCR( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_ROCR_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ROCR( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ROCR100( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_ROCR100_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ROCR100( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def RSI( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_RSI_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_RSI( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def SAR( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , optInAcceleration , optInMaximum ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_SAR_Lookback( optInAcceleration , optInMaximum )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_SAR( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , optInAcceleration , optInMaximum , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def SAREXT( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , optInStartValue , optInOffsetOnReverse , optInAccelerationInitLong , optInAccelerationLong , optInAccelerationMaxLong , optInAccelerationInitShort , optInAccelerationShort , optInAccelerationMaxShort ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_SAREXT_Lookback( optInStartValue , optInOffsetOnReverse , optInAccelerationInitLong , optInAccelerationLong , optInAccelerationMaxLong , optInAccelerationInitShort , optInAccelerationShort , optInAccelerationMaxShort )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_SAREXT( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , optInStartValue , optInOffsetOnReverse , optInAccelerationInitLong , optInAccelerationLong , optInAccelerationMaxLong , optInAccelerationInitShort , optInAccelerationShort , optInAccelerationMaxShort , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def SIN( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_SIN_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_SIN( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def SINH( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_SINH_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_SINH( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def SMA( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_SMA_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_SMA( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def SQRT( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_SQRT_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_SQRT( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def STDDEV( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 , optInNbDev=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_STDDEV_Lookback( optInTimePeriod , optInNbDev )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_STDDEV( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , optInNbDev , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def STOCH( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInFastK_Period=1 , optInSlowK_Period=1 , optInSlowK_MAType=1 , optInSlowD_Period=1 , optInSlowD_MAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_STOCH_Lookback( optInFastK_Period , optInSlowK_Period , optInSlowK_MAType , optInSlowD_Period , optInSlowD_MAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outSlowK = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outSlowD = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_STOCH( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInFastK_Period , optInSlowK_Period , optInSlowK_MAType , optInSlowD_Period , optInSlowD_MAType , &outBegIdx , &outNBElement , <double *>outSlowK.data , <double *>outSlowD.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outSlowK , outSlowD )

def STOCHF( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInFastK_Period=1 , optInFastD_Period=1 , optInFastD_MAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_STOCHF_Lookback( optInFastK_Period , optInFastD_Period , optInFastD_MAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outFastK = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outFastD = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_STOCHF( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInFastK_Period , optInFastD_Period , optInFastD_MAType , &outBegIdx , &outNBElement , <double *>outFastK.data , <double *>outFastD.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outFastK , outFastD )

def STOCHRSI( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 , optInFastK_Period=1 , optInFastD_Period=1 , optInFastD_MAType=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_STOCHRSI_Lookback( optInTimePeriod , optInFastK_Period , optInFastD_Period , optInFastD_MAType )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outFastK = numpy.zeros(allocationSize)
    cdef np.ndarray[np.float_t, ndim=1] outFastD = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_STOCHRSI( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , optInFastK_Period , optInFastD_Period , optInFastD_MAType , &outBegIdx , &outNBElement , <double *>outFastK.data , <double *>outFastD.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outFastK , outFastD )

def SUB( np.ndarray[np.float_t, ndim=1] inReal0 , np.ndarray[np.float_t, ndim=1] inReal1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal0.shape[0] - 1
    cdef int lookback = TA_SUB_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_SUB( startIdx , endIdx , <double *>inReal0.data , <double *>inReal1.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def SUM( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_SUM_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_SUM( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def T3( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 , optInVFactor=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_T3_Lookback( optInTimePeriod , optInVFactor )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_T3( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , optInVFactor , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def TAN( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_TAN_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_TAN( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def TANH( np.ndarray[np.float_t, ndim=1] inReal ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_TANH_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_TANH( startIdx , endIdx , <double *>inReal.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def TEMA( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_TEMA_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_TEMA( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def TRANGE( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_TRANGE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_TRANGE( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def TRIMA( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_TRIMA_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_TRIMA( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def TRIX( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_TRIX_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_TRIX( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def TSF( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_TSF_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_TSF( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def TYPPRICE( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_TYPPRICE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_TYPPRICE( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def ULTOSC( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod1 , optInTimePeriod2 , optInTimePeriod3 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_ULTOSC_Lookback( optInTimePeriod1 , optInTimePeriod2 , optInTimePeriod3 )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_ULTOSC( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod1 , optInTimePeriod2 , optInTimePeriod3 , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def VAR( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 , optInNbDev=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_VAR_Lookback( optInTimePeriod , optInNbDev )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_VAR( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , optInNbDev , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def WCLPRICE( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_WCLPRICE_Lookback( )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_WCLPRICE( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def WILLR( np.ndarray[np.float_t, ndim=1] inHigh , np.ndarray[np.float_t, ndim=1] inLow , np.ndarray[np.float_t, ndim=1] inClose , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inHigh.shape[0] - 1
    cdef int lookback = TA_WILLR_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_WILLR( startIdx , endIdx , <double *>inHigh.data , <double *>inLow.data , <double *>inClose.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

def WMA( np.ndarray[np.float_t, ndim=1] inReal , optInTimePeriod=1 ):
    cdef int startIdx = 0
    cdef int endIdx = inReal.shape[0] - 1
    cdef int lookback = TA_WMA_Lookback( optInTimePeriod )
    cdef int temp = max(lookback, startIdx )
    cdef int allocationSize
    if ( temp > endIdx ):
        allocationSize = 0
    else:
        allocationSize = endIdx - temp + 1
    cdef int outBegIdx
    cdef int outNBElement
    cdef np.ndarray[np.float_t, ndim=1] outReal = numpy.zeros(allocationSize)
    retCode = TA_Initialize()
    if retCode != TA_SUCCESS:
        raise Exception("Cannot initialize TA-Lib (%d)!" % retCode)
    else:
        retCode = TA_WMA( startIdx , endIdx , <double *>inReal.data , optInTimePeriod , &outBegIdx , &outNBElement , <double *>outReal.data )
    TA_Shutdown()
    return ( outBegIdx , outNBElement , outReal )

