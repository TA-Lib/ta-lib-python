# Microsoft Developer Studio Project File - Name="ta_regtest" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=TA_REGTEST - WIN32 CMD MULTITHREAD DEBUG
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ta_regtest.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ta_regtest.mak" CFG="TA_REGTEST - WIN32 CMD MULTITHREAD DEBUG"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ta_regtest - Win32 CMD Multithread Debug" (based on "Win32 (x86) Console Application")
!MESSAGE "ta_regtest - Win32 CSD Single Thread Debug" (based on "Win32 (x86) Console Application")
!MESSAGE "ta_regtest - Win32 CSR Single Thread Release" (based on "Win32 (x86) Console Application")
!MESSAGE "ta_regtest - Win32 CMR Multithread Release" (based on "Win32 (x86) Console Application")
!MESSAGE "ta_regtest - Win32 Profiling" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ta_regtest - Win32 CMD Multithread Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "ta_regtest___Win32_CMD_Multithread_Debug"
# PROP BASE Intermediate_Dir "ta_regtest___Win32_CMD_Multithread_Debug"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "..\..\..\..\bin"
# PROP Intermediate_Dir "..\..\..\..\temp\cmd\ta_regtest"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MDd /W3 /Gm /Zi /Od /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /D "_CONSOLE" /D "TA_DEBUG" /D "WIN32" /D "_DEBUG" /D "_MBCS" /FD /GZ /c
# SUBTRACT BASE CPP /Fr /YX
# ADD CPP /nologo /MTd /W3 /WX /Gm /Zi /Od /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /I "..\..\..\..\src\ta_common\mt" /D "_CONSOLE" /D "TA_DEBUG" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "TA_FUNC_NO_RANGE_CHECK" /Fr /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 ta_libc_cmd.lib wininet.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /incremental:no /debug /debugtype:both /machine:I386 /libpath:"..\..\..\..\lib"
# SUBTRACT BASE LINK32 /profile
# ADD LINK32 ta_libc_cmd.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:console /incremental:no /debug /debugtype:both /machine:I386 /libpath:"..\..\..\..\lib"
# SUBTRACT LINK32 /profile

!ELSEIF  "$(CFG)" == "ta_regtest - Win32 CSD Single Thread Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "ta_regtest___Win32_CSD_Single_Thread_Debug"
# PROP BASE Intermediate_Dir "ta_regtest___Win32_CSD_Single_Thread_Debug"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "..\..\..\..\bin"
# PROP Intermediate_Dir "..\..\..\..\temp\csd\ta_regtest"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /Zi /Od /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /D "_CONSOLE" /D "TA_DEBUG" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "TA_SINGLE_THREAD" /FD /GZ /c
# SUBTRACT BASE CPP /Fr /YX
# ADD CPP /nologo /W3 /WX /Gm /Zi /Od /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /I "..\..\..\..\src\ta_common\mt" /D "_CONSOLE" /D "TA_DEBUG" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "TA_SINGLE_THREAD" /Fr /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 ta_libc_csd.lib wininet.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /profile /debug /debugtype:both /machine:I386 /libpath:"..\..\..\..\lib"
# ADD LINK32 ta_libc_csd.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:console /profile /debug /debugtype:both /machine:I386 /libpath:"..\..\..\..\lib"

!ELSEIF  "$(CFG)" == "ta_regtest - Win32 CSR Single Thread Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_regtest___Win32_CSR_Single_Thread_Release"
# PROP BASE Intermediate_Dir "ta_regtest___Win32_CSR_Single_Thread_Release"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\bin"
# PROP Intermediate_Dir "..\..\..\..\temp\csr\ta_regtest"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /O2 /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /D "_CONSOLE" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /FD /c
# SUBTRACT BASE CPP /Fr /YX
# ADD CPP /nologo /W3 /WX /O2 /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /I "..\..\..\..\src\ta_common\mt" /D "_CONSOLE" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /FD /c
# SUBTRACT CPP /Fr /YX
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 ta_libc_csr.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib wininet.lib /nologo /subsystem:console /machine:I386 /libpath:"..\..\..\..\lib"
# ADD LINK32 ta_libc_csr.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:console /machine:I386 /libpath:"..\..\..\..\lib"

!ELSEIF  "$(CFG)" == "ta_regtest - Win32 CMR Multithread Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_regtest___Win32_CMR_Multithread_Release0"
# PROP BASE Intermediate_Dir "ta_regtest___Win32_CMR_Multithread_Release0"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\bin"
# PROP Intermediate_Dir "..\..\..\..\temp\cmr\ta_regtest"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MD /W3 /O1 /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /D "_CONSOLE" /D "WIN32" /D "NDEBUG" /D "_MBCS" /FD /c
# SUBTRACT BASE CPP /Fr /YX
# ADD CPP /nologo /MT /W3 /WX /O2 /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /I "..\..\..\..\src\ta_common\mt" /D "_CONSOLE" /D "WIN32" /D "NDEBUG" /D "_MBCS" /FD /c
# SUBTRACT CPP /Fr /YX
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 ta_libc_cmr.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib wininet.lib /nologo /subsystem:console /machine:I386 /libpath:"..\..\..\..\lib"
# ADD LINK32 ta_libc_cmr.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:console /machine:I386 /libpath:"..\..\..\..\lib"

!ELSEIF  "$(CFG)" == "ta_regtest - Win32 Profiling"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_regtest___Win32_Profiling"
# PROP BASE Intermediate_Dir "ta_regtest___Win32_Profiling"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "ta_regtest___Win32_Profiling"
# PROP Intermediate_Dir "ta_regtest___Win32_Profiling"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /O2 /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /D "_CONSOLE" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /FD /c
# SUBTRACT BASE CPP /Fr /YX
# ADD CPP /nologo /W3 /WX /O2 /I "..\..\..\..\src\ta_common" /I "..\..\..\..\include" /I "..\..\..\..\src\ta_common\trio" /I "..\..\..\..\src\tools\ta_regtest" /I "..\..\..\..\src\ta_common\imatix\sfl" /I "..\..\..\..\src\ta_func" /I "..\..\..\..\src\ta_data\ta_source\ta_readop" /I "..\..\..\..\src\ta_common\mt" /D "_CONSOLE" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /FD /c
# SUBTRACT CPP /Fr /YX
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 ta_libc_csr.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib wininet.lib /nologo /subsystem:console /machine:I386 /libpath:"..\..\..\..\lib"
# ADD LINK32 ta_libc_csr.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:console /profile /machine:I386 /libpath:"..\..\..\..\lib"

!ENDIF 

# Begin Target

# Name "ta_regtest - Win32 CMD Multithread Debug"
# Name "ta_regtest - Win32 CSD Single Thread Debug"
# Name "ta_regtest - Win32 CSR Single Thread Release"
# Name "ta_regtest - Win32 CMR Multithread Release"
# Name "ta_regtest - Win32 Profiling"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_regtest.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_1in_1out.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_1in_2out.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\test_abstract.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_adx.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_bbands.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_candlestick.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\test_data.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\test_internals.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_ma.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_macd.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_minmax.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_mom.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_per_ema.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_per_hl.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_per_hlc.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_per_hlcv.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_per_ohlc.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_po.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_rsi.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_sar.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_stddev.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_stoch.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func\test_trange.c
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\test_util.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_error_number.h
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_func.h
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\ta_test_priv.h
# End Source File
# Begin Source File

SOURCE=..\..\..\..\src\tools\ta_regtest\test_period.h
# End Source File
# End Group
# End Target
# End Project
