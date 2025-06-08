#cython: language_level=2

cdef extern from "ta-lib/ta_defs.h":

    ctypedef int TA_RetCode
    TA_RetCode TA_SUCCESS = 0
    TA_RetCode TA_LIB_NOT_INITIALIZE = 1
    TA_RetCode TA_BAD_PARAM = 2
    TA_RetCode TA_ALLOC_ERR = 3
    TA_RetCode TA_GROUP_NOT_FOUND = 4
    TA_RetCode TA_FUNC_NOT_FOUND = 5
    TA_RetCode TA_INVALID_HANDLE = 6
    TA_RetCode TA_INVALID_PARAM_HOLDER = 7
    TA_RetCode TA_INVALID_PARAM_HOLDER_TYPE = 8
    TA_RetCode TA_INVALID_PARAM_FUNCTION = 9
    TA_RetCode TA_INPUT_NOT_ALL_INITIALIZE = 10
    TA_RetCode TA_OUTPUT_NOT_ALL_INITIALIZE = 11
    TA_RetCode TA_OUT_OF_RANGE_START_INDEX = 12
    TA_RetCode TA_OUT_OF_RANGE_END_INDEX = 13
    TA_RetCode TA_INVALID_LIST_TYPE = 14
    TA_RetCode TA_BAD_OBJECT = 15
    TA_RetCode TA_NOT_SUPPORTED = 16
    TA_RetCode TA_INTERNAL_ERROR = 5000
    TA_RetCode TA_UNKNOWN_ERR = 0xffff

    ctypedef int TA_Compatibility
    TA_Compatibility TA_COMPATIBILITY_DEFAULT = 0
    TA_Compatibility TA_COMPATIBILITY_METASTOCK = 1

    ctypedef int TA_MAType
    TA_MAType TA_MAType_SMA = 0
    TA_MAType TA_MAType_EMA = 1
    TA_MAType TA_MAType_WMA = 2
    TA_MAType TA_MAType_DEMA = 3
    TA_MAType TA_MAType_TEMA = 4
    TA_MAType TA_MAType_TRIMA = 5
    TA_MAType TA_MAType_KAMA = 6
    TA_MAType TA_MAType_MAMA = 7
    TA_MAType TA_MAType_T3 = 8

    ctypedef int TA_FuncUnstId
    TA_FuncUnstId TA_FUNC_UNST_ADX = 0
    TA_FuncUnstId TA_FUNC_UNST_ADXR = 1
    TA_FuncUnstId TA_FUNC_UNST_ATR = 2
    TA_FuncUnstId TA_FUNC_UNST_CMO = 3
    TA_FuncUnstId TA_FUNC_UNST_DX = 4
    TA_FuncUnstId TA_FUNC_UNST_EMA = 5
    TA_FuncUnstId TA_FUNC_UNST_HT_DCPERIOD = 6
    TA_FuncUnstId TA_FUNC_UNST_HT_DCPHASE = 7
    TA_FuncUnstId TA_FUNC_UNST_HD_PHASOR = 8
    TA_FuncUnstId TA_FUNC_UNST_HT_SINE = 9
    TA_FuncUnstId TA_FUNC_UNST_HT_TRENDLINE = 10
    TA_FuncUnstId TA_FUNC_UNST_HT_TRENDMODE = 11
    TA_FuncUnstId TA_FUNC_UNST_KAMA = 12
    TA_FuncUnstId TA_FUNC_UNST_MAMA = 13
    TA_FuncUnstId TA_FUNC_UNST_MFI = 14
    TA_FuncUnstId TA_FUNC_UNST_MINUS_DI = 15
    TA_FuncUnstId TA_FUNC_UNST_MINUS_DM = 16
    TA_FuncUnstId TA_FUNC_UNST_NATR = 17
    TA_FuncUnstId TA_FUNC_UNST_PLUS_DI = 18
    TA_FuncUnstId TA_FUNC_UNST_PLUS_DM = 19
    TA_FuncUnstId TA_FUNC_UNST_RSI = 20
    TA_FuncUnstId TA_FUNC_UNST_STOCHRSI = 21
    TA_FuncUnstId TA_FUNC_UNST_T3 = 21
    TA_FuncUnstId TA_FUNC_UNST_ALL = 22
    TA_FuncUnstId TA_FUNC_UNST_NONE = -1

    ctypedef int TA_RangeType
    TA_RangeType TA_RangeType_RealBody = 0
    TA_RangeType TA_RangeType_HighLow = 1
    TA_RangeType TA_RangeType_Shadows = 2

    ctypedef int TA_CandleSettingType
    TA_CandleSettingType TA_BodyLong = 0
    TA_CandleSettingType TA_BodyVeryLong = 1
    TA_CandleSettingType TA_BodyShort = 2
    TA_CandleSettingType TA_BodyDoji = 3
    TA_CandleSettingType TA_ShadowLong = 4
    TA_CandleSettingType TA_ShadowVeryLong = 5
    TA_CandleSettingType TA_ShadowShort = 6
    TA_CandleSettingType TA_ShadowVeryShort = 7
    TA_CandleSettingType TA_Near = 8
    TA_CandleSettingType TA_Far = 9
    TA_CandleSettingType TA_Equal = 10
    TA_CandleSettingType TA_AllCandleSettings = 11

cdef extern from "ta-lib/ta_common.h":
    const char *TA_GetVersionString()
    const char *TA_GetVersionMajor()
    const char *TA_GetVersionMinor()
    const char *TA_GetVersionBuild()
    const char *TA_GetVersionDate()
    const char *TA_GetVersionTime()

    ctypedef double TA_Real
    ctypedef int TA_Integer

    ctypedef struct TA_StringTable:
        unsigned int size
        const char **string
        void *hiddenData

    ctypedef struct TA_RetCodeInfo:
        const char* enumStr
        const char* infoStr

    void TA_SetRetCodeInfo(TA_RetCode theRetCode, TA_RetCodeInfo *retCodeInfo)

    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()

cdef extern from "ta-lib/ta_abstract.h":

    TA_RetCode TA_GroupTableAlloc(TA_StringTable **table)
    TA_RetCode TA_GroupTableFree(TA_StringTable *table)

    TA_RetCode TA_FuncTableAlloc(const char *group, TA_StringTable **table)
    TA_RetCode TA_FuncTableFree(TA_StringTable *table)

    ctypedef unsigned int TA_FuncHandle
    TA_RetCode TA_GetFuncHandle(const char *name, const TA_FuncHandle **handle)

    ctypedef int TA_FuncFlags
    ctypedef struct TA_FuncInfo:
        const char *name
        const char *group
        const char *hint
        const char *camelCaseName
        TA_FuncFlags flags
        unsigned int nbInput
        unsigned int nbOptInput
        unsigned int nbOutput
        const TA_FuncHandle *handle

    TA_RetCode TA_GetFuncInfo(const TA_FuncHandle *handle, const TA_FuncInfo **funcInfo)

    ctypedef int TA_InputParameterType
    TA_InputParameterType TA_Input_Price = 0
    TA_InputParameterType TA_Input_Real = 1
    TA_InputParameterType TA_Input_Integer = 2

    ctypedef int TA_OptInputParameterType
    TA_OptInputParameterType TA_OptInput_RealRange = 0
    TA_OptInputParameterType TA_OptInput_RealList = 1
    TA_OptInputParameterType TA_OptInput_IntegerRange = 2
    TA_OptInputParameterType TA_OptInput_IntegerList = 3

    ctypedef int TA_OutputParameterType
    TA_OutputParameterType TA_Output_Real = 0
    TA_OutputParameterType TA_Output_Integer = 1

    ctypedef int TA_InputFlags
    ctypedef int TA_OptInputFlags
    ctypedef int TA_OutputFlags

    ctypedef struct TA_InputParameterInfo:
        TA_InputParameterType type
        const char *paramName
        TA_InputFlags flags

    ctypedef struct TA_OptInputParameterInfo:
        TA_OptInputParameterType type
        const char *paramName
        TA_OptInputFlags flags
        const char *displayName
        const void *dataSet
        TA_Real defaultValue
        const char *hint
        const char *helpFile

    ctypedef struct TA_OutputParameterInfo:
        TA_OutputParameterType type
        const char *paramName
        TA_OutputFlags flags

    TA_RetCode TA_GetInputParameterInfo(const TA_FuncHandle *handle, unsigned int paramIndex, const TA_InputParameterInfo **info)
    TA_RetCode TA_GetOptInputParameterInfo(const TA_FuncHandle *handle, unsigned int paramIndex, const TA_OptInputParameterInfo **info)
    TA_RetCode TA_GetOutputParameterInfo(const TA_FuncHandle *handle, unsigned int paramIndex, const TA_OutputParameterInfo **info)

    ctypedef struct TA_ParamHolder:
        void *hiddenData

    TA_RetCode TA_ParamHolderAlloc(const TA_FuncHandle *handle, TA_ParamHolder **allocatedParams) # get_lookback()
    TA_RetCode TA_ParamHolderFree(TA_ParamHolder *params)

    TA_RetCode TA_SetOptInputParamInteger(TA_ParamHolder *params, unsigned int paramIndex, TA_Integer optInValue)
    TA_RetCode TA_SetOptInputParamReal(TA_ParamHolder *params, unsigned int paramIndex, TA_Real optInValue)

    TA_RetCode TA_GetLookback(const TA_ParamHolder *params, TA_Integer *lookback)

    TA_RetCode TA_CallFunc(const TA_ParamHolder *params, TA_Integer startIdx, TA_Integer endIdx, TA_Integer *outBegIdx, TA_Integer *outNbElement)

    char* TA_FunctionDescriptionXML()

cdef extern from "ta-lib/ta_func.h":
    TA_RetCode TA_ACCBANDS(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outRealUpperBand[], double outRealMiddleBand[], double outRealLowerBand[])
    int TA_ACCBANDS_Lookback(int optInTimePeriod)
    TA_RetCode TA_ACOS(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ACOS_Lookback()
    TA_RetCode TA_AD(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], const double inVolume[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_AD_Lookback()
    TA_RetCode TA_ADD(int startIdx, int endIdx, const double inReal0[], const double inReal1[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ADD_Lookback()
    TA_RetCode TA_ADOSC(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], const double inVolume[], int optInFastPeriod, int optInSlowPeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ADOSC_Lookback(int optInFastPeriod, int optInSlowPeriod)
    TA_RetCode TA_ADX(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ADX_Lookback(int optInTimePeriod)
    TA_RetCode TA_ADXR(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ADXR_Lookback(int optInTimePeriod)
    TA_RetCode TA_APO(int startIdx, int endIdx, const double inReal[], int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_APO_Lookback(int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType)
    TA_RetCode TA_AROON(int startIdx, int endIdx, const double inHigh[], const double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outAroonDown[], double outAroonUp[])
    int TA_AROON_Lookback(int optInTimePeriod)
    TA_RetCode TA_AROONOSC(int startIdx, int endIdx, const double inHigh[], const double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_AROONOSC_Lookback(int optInTimePeriod)
    TA_RetCode TA_ASIN(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ASIN_Lookback()
    TA_RetCode TA_ATAN(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ATAN_Lookback()
    TA_RetCode TA_ATR(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ATR_Lookback(int optInTimePeriod)
    TA_RetCode TA_AVGPRICE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_AVGPRICE_Lookback()
    TA_RetCode TA_AVGDEV(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_AVGDEV_Lookback(int optInTimePeriod)
    TA_RetCode TA_BBANDS(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, double optInNbDevUp, double optInNbDevDn, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outRealUpperBand[], double outRealMiddleBand[], double outRealLowerBand[])
    int TA_BBANDS_Lookback(int optInTimePeriod, double optInNbDevUp, double optInNbDevDn, TA_MAType optInMAType)
    TA_RetCode TA_BETA(int startIdx, int endIdx, const double inReal0[], const double inReal1[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_BETA_Lookback(int optInTimePeriod)
    TA_RetCode TA_BOP(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_BOP_Lookback()
    TA_RetCode TA_CCI(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_CCI_Lookback(int optInTimePeriod)
    TA_RetCode TA_CDL2CROWS(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDL2CROWS_Lookback()
    TA_RetCode TA_CDL3BLACKCROWS(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDL3BLACKCROWS_Lookback()
    TA_RetCode TA_CDL3INSIDE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDL3INSIDE_Lookback()
    TA_RetCode TA_CDL3LINESTRIKE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDL3LINESTRIKE_Lookback()
    TA_RetCode TA_CDL3OUTSIDE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDL3OUTSIDE_Lookback()
    TA_RetCode TA_CDL3STARSINSOUTH(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDL3STARSINSOUTH_Lookback()
    TA_RetCode TA_CDL3WHITESOLDIERS(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDL3WHITESOLDIERS_Lookback()
    TA_RetCode TA_CDLABANDONEDBABY(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLABANDONEDBABY_Lookback(double optInPenetration)
    TA_RetCode TA_CDLADVANCEBLOCK(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLADVANCEBLOCK_Lookback()
    TA_RetCode TA_CDLBELTHOLD(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLBELTHOLD_Lookback()
    TA_RetCode TA_CDLBREAKAWAY(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLBREAKAWAY_Lookback()
    TA_RetCode TA_CDLCLOSINGMARUBOZU(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLCLOSINGMARUBOZU_Lookback()
    TA_RetCode TA_CDLCONCEALBABYSWALL(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLCONCEALBABYSWALL_Lookback()
    TA_RetCode TA_CDLCOUNTERATTACK(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLCOUNTERATTACK_Lookback()
    TA_RetCode TA_CDLDARKCLOUDCOVER(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLDARKCLOUDCOVER_Lookback(double optInPenetration)
    TA_RetCode TA_CDLDOJI(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLDOJI_Lookback()
    TA_RetCode TA_CDLDOJISTAR(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLDOJISTAR_Lookback()
    TA_RetCode TA_CDLDRAGONFLYDOJI(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLDRAGONFLYDOJI_Lookback()
    TA_RetCode TA_CDLENGULFING(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLENGULFING_Lookback()
    TA_RetCode TA_CDLEVENINGDOJISTAR(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLEVENINGDOJISTAR_Lookback(double optInPenetration)
    TA_RetCode TA_CDLEVENINGSTAR(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLEVENINGSTAR_Lookback(double optInPenetration)
    TA_RetCode TA_CDLGAPSIDESIDEWHITE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLGAPSIDESIDEWHITE_Lookback()
    TA_RetCode TA_CDLGRAVESTONEDOJI(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLGRAVESTONEDOJI_Lookback()
    TA_RetCode TA_CDLHAMMER(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLHAMMER_Lookback()
    TA_RetCode TA_CDLHANGINGMAN(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLHANGINGMAN_Lookback()
    TA_RetCode TA_CDLHARAMI(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLHARAMI_Lookback()
    TA_RetCode TA_CDLHARAMICROSS(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLHARAMICROSS_Lookback()
    TA_RetCode TA_CDLHIGHWAVE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLHIGHWAVE_Lookback()
    TA_RetCode TA_CDLHIKKAKE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLHIKKAKE_Lookback()
    TA_RetCode TA_CDLHIKKAKEMOD(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLHIKKAKEMOD_Lookback()
    TA_RetCode TA_CDLHOMINGPIGEON(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLHOMINGPIGEON_Lookback()
    TA_RetCode TA_CDLIDENTICAL3CROWS(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLIDENTICAL3CROWS_Lookback()
    TA_RetCode TA_CDLINNECK(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLINNECK_Lookback()
    TA_RetCode TA_CDLINVERTEDHAMMER(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLINVERTEDHAMMER_Lookback()
    TA_RetCode TA_CDLKICKING(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLKICKING_Lookback()
    TA_RetCode TA_CDLKICKINGBYLENGTH(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLKICKINGBYLENGTH_Lookback()
    TA_RetCode TA_CDLLADDERBOTTOM(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLLADDERBOTTOM_Lookback()
    TA_RetCode TA_CDLLONGLEGGEDDOJI(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLLONGLEGGEDDOJI_Lookback()
    TA_RetCode TA_CDLLONGLINE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLLONGLINE_Lookback()
    TA_RetCode TA_CDLMARUBOZU(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLMARUBOZU_Lookback()
    TA_RetCode TA_CDLMATCHINGLOW(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLMATCHINGLOW_Lookback()
    TA_RetCode TA_CDLMATHOLD(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLMATHOLD_Lookback(double optInPenetration)
    TA_RetCode TA_CDLMORNINGDOJISTAR(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLMORNINGDOJISTAR_Lookback(double optInPenetration)
    TA_RetCode TA_CDLMORNINGSTAR(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], double optInPenetration, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLMORNINGSTAR_Lookback(double optInPenetration)
    TA_RetCode TA_CDLONNECK(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLONNECK_Lookback()
    TA_RetCode TA_CDLPIERCING(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLPIERCING_Lookback()
    TA_RetCode TA_CDLRICKSHAWMAN(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLRICKSHAWMAN_Lookback()
    TA_RetCode TA_CDLRISEFALL3METHODS(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLRISEFALL3METHODS_Lookback()
    TA_RetCode TA_CDLSEPARATINGLINES(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLSEPARATINGLINES_Lookback()
    TA_RetCode TA_CDLSHOOTINGSTAR(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLSHOOTINGSTAR_Lookback()
    TA_RetCode TA_CDLSHORTLINE(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLSHORTLINE_Lookback()
    TA_RetCode TA_CDLSPINNINGTOP(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLSPINNINGTOP_Lookback()
    TA_RetCode TA_CDLSTALLEDPATTERN(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLSTALLEDPATTERN_Lookback()
    TA_RetCode TA_CDLSTICKSANDWICH(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLSTICKSANDWICH_Lookback()
    TA_RetCode TA_CDLTAKURI(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLTAKURI_Lookback()
    TA_RetCode TA_CDLTASUKIGAP(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLTASUKIGAP_Lookback()
    TA_RetCode TA_CDLTHRUSTING(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLTHRUSTING_Lookback()
    TA_RetCode TA_CDLTRISTAR(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLTRISTAR_Lookback()
    TA_RetCode TA_CDLUNIQUE3RIVER(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLUNIQUE3RIVER_Lookback()
    TA_RetCode TA_CDLUPSIDEGAP2CROWS(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLUPSIDEGAP2CROWS_Lookback()
    TA_RetCode TA_CDLXSIDEGAP3METHODS(int startIdx, int endIdx, const double inOpen[], const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_CDLXSIDEGAP3METHODS_Lookback()
    TA_RetCode TA_CEIL(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_CEIL_Lookback()
    TA_RetCode TA_CMO(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_CMO_Lookback(int optInTimePeriod)
    TA_RetCode TA_CORREL(int startIdx, int endIdx, const double inReal0[], const double inReal1[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_CORREL_Lookback(int optInTimePeriod)
    TA_RetCode TA_COS(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_COS_Lookback()
    TA_RetCode TA_COSH(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_COSH_Lookback()
    TA_RetCode TA_DEMA(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_DEMA_Lookback(int optInTimePeriod)
    TA_RetCode TA_DIV(int startIdx, int endIdx, const double inReal0[], const double inReal1[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_DIV_Lookback()
    TA_RetCode TA_DX(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_DX_Lookback(int optInTimePeriod)
    TA_RetCode TA_EMA(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_EMA_Lookback(int optInTimePeriod)
    TA_RetCode TA_EXP(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_EXP_Lookback()
    TA_RetCode TA_FLOOR(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_FLOOR_Lookback()
    TA_RetCode TA_HT_DCPERIOD(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_HT_DCPERIOD_Lookback()
    TA_RetCode TA_HT_DCPHASE(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_HT_DCPHASE_Lookback()
    TA_RetCode TA_HT_PHASOR(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outInPhase[], double outQuadrature[])
    int TA_HT_PHASOR_Lookback()
    TA_RetCode TA_HT_SINE(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outSine[], double outLeadSine[])
    int TA_HT_SINE_Lookback()
    TA_RetCode TA_HT_TRENDLINE(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_HT_TRENDLINE_Lookback()
    TA_RetCode TA_HT_TRENDMODE(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_HT_TRENDMODE_Lookback()
    TA_RetCode TA_IMI(int startIdx, int endIdx, const double inOpen[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_IMI_Lookback(int optInTimePeriod)
    TA_RetCode TA_KAMA(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_KAMA_Lookback(int optInTimePeriod)
    TA_RetCode TA_LINEARREG(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_LINEARREG_Lookback(int optInTimePeriod)
    TA_RetCode TA_LINEARREG_ANGLE(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_LINEARREG_ANGLE_Lookback(int optInTimePeriod)
    TA_RetCode TA_LINEARREG_INTERCEPT(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_LINEARREG_INTERCEPT_Lookback(int optInTimePeriod)
    TA_RetCode TA_LINEARREG_SLOPE(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_LINEARREG_SLOPE_Lookback(int optInTimePeriod)
    TA_RetCode TA_LN(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_LN_Lookback()
    TA_RetCode TA_LOG10(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_LOG10_Lookback()
    TA_RetCode TA_MA(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MA_Lookback(int optInTimePeriod, TA_MAType optInMAType)
    TA_RetCode TA_MACD(int startIdx, int endIdx, const double inReal[], int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod, int *outBegIdx, int *outNBElement, double outMACD[], double outMACDSignal[], double outMACDHist[])
    int TA_MACD_Lookback(int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod)
    TA_RetCode TA_MACDEXT(int startIdx, int endIdx, const double inReal[], int optInFastPeriod, TA_MAType optInFastMAType, int optInSlowPeriod, TA_MAType optInSlowMAType, int optInSignalPeriod, TA_MAType optInSignalMAType, int *outBegIdx, int *outNBElement, double outMACD[], double outMACDSignal[], double outMACDHist[])
    int TA_MACDEXT_Lookback(int optInFastPeriod, TA_MAType optInFastMAType, int optInSlowPeriod, TA_MAType optInSlowMAType, int optInSignalPeriod, TA_MAType optInSignalMAType)
    TA_RetCode TA_MACDFIX(int startIdx, int endIdx, const double inReal[], int optInSignalPeriod, int *outBegIdx, int *outNBElement, double outMACD[], double outMACDSignal[], double outMACDHist[])
    int TA_MACDFIX_Lookback(int optInSignalPeriod)
    TA_RetCode TA_MAMA(int startIdx, int endIdx, const double inReal[], double optInFastLimit, double optInSlowLimit, int *outBegIdx, int *outNBElement, double outMAMA[], double outFAMA[])
    int TA_MAMA_Lookback(double optInFastLimit, double optInSlowLimit)
    TA_RetCode TA_MAVP(int startIdx, int endIdx, const double inReal[], const double inPeriods[], int optInMinPeriod, int optInMaxPeriod, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MAVP_Lookback(int optInMinPeriod, int optInMaxPeriod, TA_MAType optInMAType)
    TA_RetCode TA_MAX(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MAX_Lookback(int optInTimePeriod)
    TA_RetCode TA_MAXINDEX(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_MAXINDEX_Lookback(int optInTimePeriod)
    TA_RetCode TA_MEDPRICE(int startIdx, int endIdx, const double inHigh[], const double inLow[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MEDPRICE_Lookback()
    TA_RetCode TA_MFI(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], const double inVolume[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MFI_Lookback(int optInTimePeriod)
    TA_RetCode TA_MIDPOINT(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MIDPOINT_Lookback(int optInTimePeriod)
    TA_RetCode TA_MIDPRICE(int startIdx, int endIdx, const double inHigh[], const double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MIDPRICE_Lookback(int optInTimePeriod)
    TA_RetCode TA_MIN(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MIN_Lookback(int optInTimePeriod)
    TA_RetCode TA_MININDEX(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, int outInteger[])
    int TA_MININDEX_Lookback(int optInTimePeriod)
    TA_RetCode TA_MINMAX(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outMin[], double outMax[])
    int TA_MINMAX_Lookback(int optInTimePeriod)
    TA_RetCode TA_MINMAXINDEX(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, int outMinIdx[], int outMaxIdx[])
    int TA_MINMAXINDEX_Lookback(int optInTimePeriod)
    TA_RetCode TA_MINUS_DI(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MINUS_DI_Lookback(int optInTimePeriod)
    TA_RetCode TA_MINUS_DM(int startIdx, int endIdx, const double inHigh[], const double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MINUS_DM_Lookback(int optInTimePeriod)
    TA_RetCode TA_MOM(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MOM_Lookback(int optInTimePeriod)
    TA_RetCode TA_MULT(int startIdx, int endIdx, const double inReal0[], const double inReal1[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_MULT_Lookback()
    TA_RetCode TA_NATR(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_NATR_Lookback(int optInTimePeriod)
    TA_RetCode TA_OBV(int startIdx, int endIdx, const double inReal[], const double inVolume[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_OBV_Lookback()
    TA_RetCode TA_PLUS_DI(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_PLUS_DI_Lookback(int optInTimePeriod)
    TA_RetCode TA_PLUS_DM(int startIdx, int endIdx, const double inHigh[], const double inLow[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_PLUS_DM_Lookback(int optInTimePeriod)
    TA_RetCode TA_PPO(int startIdx, int endIdx, const double inReal[], int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_PPO_Lookback(int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType)
    TA_RetCode TA_ROC(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ROC_Lookback(int optInTimePeriod)
    TA_RetCode TA_ROCP(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ROCP_Lookback(int optInTimePeriod)
    TA_RetCode TA_ROCR(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ROCR_Lookback(int optInTimePeriod)
    TA_RetCode TA_ROCR100(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ROCR100_Lookback(int optInTimePeriod)
    TA_RetCode TA_RSI(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[]) nogil
    int TA_RSI_Lookback(int optInTimePeriod) nogil
    TA_RetCode TA_SAR(int startIdx, int endIdx, const double inHigh[], const double inLow[], double optInAcceleration, double optInMaximum, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_SAR_Lookback(double optInAcceleration, double optInMaximum)
    TA_RetCode TA_SAREXT(int startIdx, int endIdx, const double inHigh[], const double inLow[], double optInStartValue, double optInOffsetOnReverse, double optInAccelerationInitLong, double optInAccelerationLong, double optInAccelerationMaxLong, double optInAccelerationInitShort, double optInAccelerationShort, double optInAccelerationMaxShort, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_SAREXT_Lookback(double optInStartValue, double optInOffsetOnReverse, double optInAccelerationInitLong, double optInAccelerationLong, double optInAccelerationMaxLong, double optInAccelerationInitShort, double optInAccelerationShort, double optInAccelerationMaxShort)
    TA_RetCode TA_SIN(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_SIN_Lookback()
    TA_RetCode TA_SINH(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_SINH_Lookback()
    TA_RetCode TA_SMA(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_SMA_Lookback(int optInTimePeriod)
    TA_RetCode TA_SQRT(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_SQRT_Lookback()
    TA_RetCode TA_STDDEV(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, double optInNbDev, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_STDDEV_Lookback(int optInTimePeriod, double optInNbDev)
    TA_RetCode TA_STOCH(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInFastK_Period, int optInSlowK_Period, TA_MAType optInSlowK_MAType, int optInSlowD_Period, TA_MAType optInSlowD_MAType, int *outBegIdx, int *outNBElement, double outSlowK[], double outSlowD[])
    int TA_STOCH_Lookback(int optInFastK_Period, int optInSlowK_Period, TA_MAType optInSlowK_MAType, int optInSlowD_Period, TA_MAType optInSlowD_MAType)
    TA_RetCode TA_STOCHF(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType, int *outBegIdx, int *outNBElement, double outFastK[], double outFastD[])
    int TA_STOCHF_Lookback(int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType)
    TA_RetCode TA_STOCHRSI(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType, int *outBegIdx, int *outNBElement, double outFastK[], double outFastD[])
    int TA_STOCHRSI_Lookback(int optInTimePeriod, int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType)
    TA_RetCode TA_SUB(int startIdx, int endIdx, const double inReal0[], const double inReal1[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_SUB_Lookback()
    TA_RetCode TA_SUM(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_SUM_Lookback(int optInTimePeriod)
    TA_RetCode TA_T3(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, double optInVFactor, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_T3_Lookback(int optInTimePeriod, double optInVFactor)
    TA_RetCode TA_TAN(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_TAN_Lookback()
    TA_RetCode TA_TANH(int startIdx, int endIdx, const double inReal[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_TANH_Lookback()
    TA_RetCode TA_TEMA(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_TEMA_Lookback(int optInTimePeriod)
    TA_RetCode TA_TRANGE(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_TRANGE_Lookback()
    TA_RetCode TA_TRIMA(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_TRIMA_Lookback(int optInTimePeriod)
    TA_RetCode TA_TRIX(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_TRIX_Lookback(int optInTimePeriod)
    TA_RetCode TA_TSF(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_TSF_Lookback(int optInTimePeriod)
    TA_RetCode TA_TYPPRICE(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_TYPPRICE_Lookback()
    TA_RetCode TA_ULTOSC(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod1, int optInTimePeriod2, int optInTimePeriod3, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_ULTOSC_Lookback(int optInTimePeriod1, int optInTimePeriod2, int optInTimePeriod3)
    TA_RetCode TA_VAR(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, double optInNbDev, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_VAR_Lookback(int optInTimePeriod, double optInNbDev)
    TA_RetCode TA_WCLPRICE(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int *outBegIdx, int *outNBElement, double outReal[])
    int TA_WCLPRICE_Lookback()
    TA_RetCode TA_WILLR(int startIdx, int endIdx, const double inHigh[], const double inLow[], const double inClose[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_WILLR_Lookback(int optInTimePeriod)
    TA_RetCode TA_WMA(int startIdx, int endIdx, const double inReal[], int optInTimePeriod, int *outBegIdx, int *outNBElement, double outReal[])
    int TA_WMA_Lookback(int optInTimePeriod)

    # TALIB functions for TA_SetUnstablePeriod
    TA_RetCode TA_SetUnstablePeriod(TA_FuncUnstId id, unsigned int unstablePeriod)
    unsigned int TA_GetUnstablePeriod(TA_FuncUnstId id)

    # TALIB functions for TA_SetCompatibility
    TA_RetCode TA_SetCompatibility(TA_Compatibility value)
    TA_Compatibility TA_GetCompatibility()

    # TALIB functions for TA_SetCandleSettings
    TA_RetCode TA_SetCandleSettings(TA_CandleSettingType settingType, TA_RangeType rangeType, int avgPeriod, double factor)
    TA_RetCode TA_RestoreCandleDefaultSettings(TA_CandleSettingType)
