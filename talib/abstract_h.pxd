cdef extern from "ta-lib/ta_defs.h":
    cdef enum TA_RetCode:
        TA_SUCCESS
        TA_LIB_NOT_INITIALIZE
        TA_BAD_PARAM
        TA_ALLOC_ERR
        TA_GROUP_NOT_FOUND
        TA_FUNC_NOT_FOUND
        TA_INVALID_HANDLE
        TA_INVALID_PARAM_HOLDER
        TA_INVALID_PARAM_HOLDER_TYPE
        TA_INVALID_PARAM_FUNCTION
        TA_INPUT_NOT_ALL_INITIALIZE
        TA_OUTPUT_NOT_ALL_INITIALIZE
        TA_OUT_OF_RANGE_START_INDEX
        TA_OUT_OF_RANGE_END_INDEX
        TA_INVALID_LIST_TYPE
        TA_BAD_OBJECT
        TA_NOT_SUPPORTED
        TA_INTERNAL_ERROR
        TA_UNKNOWN_ERR

cdef extern from "ta-lib/ta_common.h":
    ctypedef int TA_Integer
    ctypedef double TA_Real
    cdef struct TA_StringTable:
        unsigned int size
        char **string
        void *hiddenData
    ctypedef TA_StringTable TA_StringTable

    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()

cdef extern from "ta-lib/ta_abstract.h":
    ctypedef int TA_FuncFlags
    ctypedef unsigned int TA_FuncHandle
    cdef struct TA_FuncInfo:
        char *name
        char *group
        char *hint
        char *camelCaseName
        TA_FuncFlags flags
        unsigned int nbInput
        unsigned int nbOptInput
        unsigned int nbOutput
        TA_FuncHandle *handle
    ctypedef TA_FuncInfo TA_FuncInfo

    ctypedef int TA_InputFlags
    ctypedef int TA_OptInputFlags
    ctypedef int TA_OutputFlags

    cdef enum TA_InputParameterType:
        TA_Input_Price
        TA_Input_Real
        TA_Input_Integer

    cdef enum TA_OptInputParameterType:
        TA_OptInput_RealRange
        TA_OptInput_RealList
        TA_OptInput_IntegerRange
        TA_OptInput_IntegerList

    cdef enum TA_OutputParameterType:
        TA_Output_Real
        TA_Output_Integer

    cdef struct TA_ParamHolder:
        void *hiddenData
    ctypedef TA_ParamHolder TA_ParamHolder

    cdef struct TA_InputParameterInfo:
        TA_InputParameterType type
        char *paramName
        TA_InputFlags flags
    ctypedef TA_InputParameterInfo TA_InputParameterInfo

    cdef struct TA_OptInputParameterInfo:
        TA_OptInputParameterType type
        char *paramName
        TA_OptInputFlags flags
        char *displayName
        void *dataSet
        TA_Real defaultValue
        char *hint
        char *helpFile
    ctypedef TA_OptInputParameterInfo TA_OptInputParameterInfo

    cdef struct TA_OutputParameterInfo:
        TA_OutputParameterType type
        char *paramName
        TA_OutputFlags flags
    ctypedef TA_OutputParameterInfo TA_OutputParameterInfo

    # TALIB Alloc/Free function pairs for getting group and function names and
    TA_RetCode TA_GroupTableAlloc(TA_StringTable **table) # used by get_functions()
    TA_RetCode TA_FuncTableAlloc(char *group, TA_StringTable **table) # get_groups()
    TA_RetCode TA_ParamHolderAlloc(TA_FuncHandle *handle, TA_ParamHolder **allocatedParams) # get_lookback()
    # be sure to call the Free functions to prevent memory leaks!
    TA_RetCode TA_GroupTableFree(TA_StringTable *table)
    TA_RetCode TA_FuncTableFree(TA_StringTable *table)
    TA_RetCode TA_ParamHolderFree(TA_ParamHolder *params)

    # TALIB get func info/handle functions
    TA_RetCode TA_GetFuncInfo(TA_FuncHandle *handle, TA_FuncInfo **funcInfo)
    TA_RetCode TA_GetFuncHandle(char *name, TA_FuncHandle **handle)
    TA_RetCode TA_GetInputParameterInfo(TA_FuncHandle *handle, unsigned int paramIndex, TA_InputParameterInfo **info)
    TA_RetCode TA_GetOptInputParameterInfo(TA_FuncHandle *handle, unsigned int paramIndex, TA_OptInputParameterInfo **info)
    TA_RetCode TA_GetOutputParameterInfo(TA_FuncHandle *handle, unsigned int paramIndex, TA_OutputParameterInfo **info)

    # TALIB set functions for TA_GetLookback/TA_CallFunc
    TA_RetCode TA_SetInputParamIntegerPtr(TA_ParamHolder *params, unsigned int paramIndex, TA_Integer *value)
    TA_RetCode TA_SetInputParamRealPtr(TA_ParamHolder *params, unsigned int paramIndex, TA_Real *value)
    TA_RetCode TA_SetInputParamPricePtr(TA_ParamHolder *params, unsigned int paramIndex, TA_Real *open, TA_Real *high, TA_Real *low, TA_Real *close, TA_Real *volume, TA_Real *openInterest)
    TA_RetCode TA_SetOptInputParamInteger(TA_ParamHolder *params, unsigned int paramIndex, TA_Integer optInValue)
    TA_RetCode TA_SetOptInputParamReal(TA_ParamHolder *params, unsigned int paramIndex, TA_Real optInValue)
    TA_RetCode TA_SetOutputParamIntegerPtr(TA_ParamHolder *params, unsigned int paramIndex, TA_Integer *out)
    TA_RetCode TA_SetOutputParamRealPtr(TA_ParamHolder *params, unsigned int paramIndex, TA_Real *out)

    # TA_GetLookback only requires TA_SetOptInputParam(s) to have been set
    TA_RetCode TA_GetLookback(TA_ParamHolder *params, TA_Integer *lookback)

    # TA_CallFunc requires all input, optionl input and output pointers to have beeen set
    TA_RetCode TA_CallFunc(TA_ParamHolder *params, TA_Integer startIdx, TA_Integer endIdx, TA_Integer *outBegIdx, TA_Integer *outNbElement)
