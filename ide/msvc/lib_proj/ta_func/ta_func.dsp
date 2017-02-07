# Microsoft Developer Studio Project File - Name="ta_func" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=ta_func - Win32 CDD Multithread DLL Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ta_func.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ta_func.mak" CFG="ta_func - Win32 CDD Multithread DLL Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ta_func - Win32 CDR Multithread DLL Release" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_func - Win32 CMD Multithread Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_func - Win32 CSD Single Thread Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_func - Win32 CSR Single Thread Release" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_func - Win32 CMR Multithread Release" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_func - Win32 Profiling" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_func - Win32 CDD Multithread DLL Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ta_func - Win32 CDR Multithread DLL Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_func___Win32_CDR_Multithread_DLL_Release"
# PROP BASE Intermediate_Dir "ta_func___Win32_CDR_Multithread_DLL_Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\cdr\ta_func"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /MD /W3 /WX /O2 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT CPP /Fr
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_func_cmr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_func_cdr.lib"

!ELSEIF  "$(CFG)" == "ta_func - Win32 CMD Multithread Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "ta_func___Win32_CMD_Multithread_Debug"
# PROP BASE Intermediate_Dir "ta_func___Win32_CMD_Multithread_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\cmd\ta_func"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /MDd /W3 /Gm /Zi /Od /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "TA_DEBUG" /D "WIN32" /D "_DEBUG" /D "_MBCS" /YX /FD /GZ /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /MTd /W3 /WX /Gm /Zi /Od /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "TA_DEBUG" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "TA_FUNC_NO_RANGE_CHECK" /Fr /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_func_cmd.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_func_cmd.lib"

!ELSEIF  "$(CFG)" == "ta_func - Win32 CSD Single Thread Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "ta_func___Win32_CSD_Single_Thread_Debug"
# PROP BASE Intermediate_Dir "ta_func___Win32_CSD_Single_Thread_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\csd\ta_func"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /W3 /Gm /GX /Zi /Od /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "TA_DEBUG" /D "TA_SINGLE_THREAD" /D "WIN32" /D "_DEBUG" /D "_MBCS" /YX /FD /GZ /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /W3 /WX /Gm /GX /Zi /Od /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "TA_DEBUG" /D "TA_SINGLE_THREAD" /D "WIN32" /D "_DEBUG" /D "_MBCS" /Fr /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_func_csd.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_func_csd.lib"

!ELSEIF  "$(CFG)" == "ta_func - Win32 CSR Single Thread Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_func___Win32_CSR_Single_Thread_Release"
# PROP BASE Intermediate_Dir "ta_func___Win32_CSR_Single_Thread_Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\csr\ta_func"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /W3 /O2 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /W3 /WX /O2 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT CPP /Fr
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_func_csr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_func_csr.lib"

!ELSEIF  "$(CFG)" == "ta_func - Win32 CMR Multithread Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_func___Win32_CMR_Multithread_Release0"
# PROP BASE Intermediate_Dir "ta_func___Win32_CMR_Multithread_Release0"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\cmr\ta_func"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /MT /W3 /WX /O2 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT CPP /Fr
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_func_cmr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_func_cmr.lib"

!ELSEIF  "$(CFG)" == "ta_func - Win32 Profiling"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_func___Win32_Profiling"
# PROP BASE Intermediate_Dir "ta_func___Win32_Profiling"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "ta_func___Win32_Profiling"
# PROP Intermediate_Dir "ta_func___Win32_Profiling"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /W3 /O2 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /W3 /WX /O2 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT CPP /Fr
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_func_csr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_func_csr.lib"

!ELSEIF  "$(CFG)" == "ta_func - Win32 CDD Multithread DLL Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_func___Win32_CDD_Multithread_DLL_Debug"
# PROP BASE Intermediate_Dir "ta_func___Win32_CDD_Multithread_DLL_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\cdd\ta_libc"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /MD /W3 /WX /O1 /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /MDd /W3 /WX /Zi /Od /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common" /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /Fr /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_func_cdr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_func_cdd.lib"

!ENDIF 

# Begin Target

# Name "ta_func - Win32 CDR Multithread DLL Release"
# Name "ta_func - Win32 CMD Multithread Debug"
# Name "ta_func - Win32 CSD Single Thread Debug"
# Name "ta_func - Win32 CSR Single Thread Release"
# Name "ta_func - Win32 CMR Multithread Release"
# Name "ta_func - Win32 Profiling"
# Name "ta_func - Win32 CDD Multithread DLL Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ACOS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_AD.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ADD.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ADOSC.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ADX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ADXR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_APO.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_AROON.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_AROONOSC.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ASIN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ATAN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ATR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_AVGPRICE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_BBANDS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_BETA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_BOP.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CCI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDL2CROWS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDL3BLACKCROWS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDL3INSIDE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDL3LINESTRIKE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDL3OUTSIDE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDL3STARSINSOUTH.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDL3WHITESOLDIERS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLABANDONEDBABY.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLADVANCEBLOCK.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLBELTHOLD.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLBREAKAWAY.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLCLOSINGMARUBOZU.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLCONCEALBABYSWALL.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLCOUNTERATTACK.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLDARKCLOUDCOVER.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLDOJI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLDOJISTAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLDRAGONFLYDOJI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLENGULFING.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLEVENINGDOJISTAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLEVENINGSTAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLGAPSIDESIDEWHITE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLGRAVESTONEDOJI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLHAMMER.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLHANGINGMAN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLHARAMI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLHARAMICROSS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLHIGHWAVE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLHIKKAKE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLHIKKAKEMOD.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLHOMINGPIGEON.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLIDENTICAL3CROWS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLINNECK.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLINVERTEDHAMMER.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLKICKING.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLKICKINGBYLENGTH.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLLADDERBOTTOM.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLLONGLEGGEDDOJI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLLONGLINE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLMARUBOZU.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLMATCHINGLOW.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLMATHOLD.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLMORNINGDOJISTAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLMORNINGSTAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLONNECK.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLPIERCING.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLRICKSHAWMAN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLRISEFALL3METHODS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLSEPARATINGLINES.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLSHOOTINGSTAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLSHORTLINE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLSPINNINGTOP.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLSTALLEDPATTERN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLSTICKSANDWICH.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLTAKURI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLTASUKIGAP.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLTHRUSTING.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLTRISTAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLUNIQUE3RIVER.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLUPSIDEGAP2CROWS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CDLXSIDEGAP3METHODS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CEIL.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CMO.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_CORREL.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_COS.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_COSH.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_DEMA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_DIV.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_DX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_EMA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_EXP.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_FLOOR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_HT_DCPERIOD.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_HT_DCPHASE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_HT_PHASOR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_HT_SINE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_HT_TRENDLINE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_HT_TRENDMODE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_KAMA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_LINEARREG.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_LINEARREG_ANGLE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_LINEARREG_INTERCEPT.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_LINEARREG_SLOPE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_LN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_LOG10.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MACD.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MACDEXT.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MACDFIX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MAMA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MAVP.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MAX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MAXINDEX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MEDPRICE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MFI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MIDPOINT.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MIDPRICE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MIN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MININDEX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MINMAX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MINMAXINDEX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MINUS_DI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MINUS_DM.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MOM.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_MULT.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_NATR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_OBV.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_PLUS_DI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_PLUS_DM.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_PPO.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ROC.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ROCP.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ROCR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ROCR100.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_RSI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_SAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_SAREXT.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_SIN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_SINH.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_SMA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_SQRT.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_STDDEV.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_STOCH.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_STOCHF.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_STOCHRSI.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_SUB.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_SUM.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_T3.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_TAN.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_TANH.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_TEMA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_TRANGE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_TRIMA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_TRIX.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_TSF.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_TYPPRICE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_ULTOSC.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_VAR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_WCLPRICE.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_WILLR.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_WMA.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_utility.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\..\..\..\src\ta_func\ta_utility.h
# End Source File
# End Group
# End Target
# End Project
