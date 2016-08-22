# Microsoft Developer Studio Project File - Name="ta_libc" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=ta_libc - Win32 CDR Multithread DLL Release
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ta_libc.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ta_libc.mak" CFG="ta_libc - Win32 CDR Multithread DLL Release"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ta_libc - Win32 CDR Multithread DLL Release" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_libc - Win32 CMD Multithread Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_libc - Win32 CSD Single Thread Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_libc - Win32 CSR Single Thread Release" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_libc - Win32 CMR Multithread Release" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_libc - Win32 Profiling" (based on "Win32 (x86) Static Library")
!MESSAGE "ta_libc - Win32 CDD Multithread DLL Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ta_libc - Win32 CDR Multithread DLL Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_libc___Win32_CDR_Multithread_DLL_Release"
# PROP BASE Intermediate_Dir "ta_libc___Win32_CDR_Multithread_DLL_Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\cdr\ta_libc"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MD /W3 /WX /O2 /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_cmr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_cdr.lib"

!ELSEIF  "$(CFG)" == "ta_libc - Win32 CMD Multithread Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "ta_libc___Win32_CMD_Multithread_Debug"
# PROP BASE Intermediate_Dir "ta_libc___Win32_CMD_Multithread_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\cmd\ta_libc"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /MDd /W3 /Gm /Zi /Od /D "_LIB" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "TA_DEBUG" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /WX /Gm /Zi /Od /D "_LIB" /D "TA_DEBUG" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "TA_FUNC_NO_RANGE_CHECK" /Fr /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_cmd.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_cmd.lib"

!ELSEIF  "$(CFG)" == "ta_libc - Win32 CSD Single Thread Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "ta_libc___Win32_CSD_Single_Thread_Debug"
# PROP BASE Intermediate_Dir "ta_libc___Win32_CSD_Single_Thread_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\csd\ta_libc"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /W3 /Gm /GX /Zi /Od /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "TA_DEBUG" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /WX /Gm /Zi /Od /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "TA_DEBUG" /FR /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_csd.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_csd.lib"

!ELSEIF  "$(CFG)" == "ta_libc - Win32 CSR Single Thread Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_libc___Win32_CSR_Single_Thread_Release"
# PROP BASE Intermediate_Dir "ta_libc___Win32_CSR_Single_Thread_Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\csr\ta_libc"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /W3 /O2 /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /WX /O2 /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_csr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_csr.lib"

!ELSEIF  "$(CFG)" == "ta_libc - Win32 CMR Multithread Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_libc___Win32_CMR_Multithread_Release"
# PROP BASE Intermediate_Dir "ta_libc___Win32_CMR_Multithread_Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\cmr\ta_libc"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /MD /W3 /O1 /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MT /W3 /WX /O2 /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# SUBTRACT CPP /Fr
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_cmr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_cmr.lib"

!ELSEIF  "$(CFG)" == "ta_libc - Win32 Profiling"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_libc___Win32_Profiling"
# PROP BASE Intermediate_Dir "ta_libc___Win32_Profiling"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "ta_libc___Win32_Profiling"
# PROP Intermediate_Dir "ta_libc___Win32_Profiling"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /W3 /O2 /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /WX /O2 /D "_LIB" /D "TA_SINGLE_THREAD" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_csr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_csr.lib"

!ELSEIF  "$(CFG)" == "ta_libc - Win32 CDD Multithread DLL Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ta_libc___Win32_CDD_Multithread_DLL_Debug"
# PROP BASE Intermediate_Dir "ta_libc___Win32_CDD_Multithread_DLL_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "..\..\..\..\lib"
# PROP Intermediate_Dir "..\..\..\..\temp\cdd\ta_libc"
# PROP Target_Dir ""
F90=df.exe
# ADD BASE CPP /nologo /MD /W3 /WX /O1 /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MDd /W3 /WX /Zi /Od /D "_LIB" /D "WIN32" /D "NDEBUG" /D "_MBCS" /FR /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_cdr.lib"
# ADD LIB32 /nologo /out:"..\..\..\..\lib\ta_libc_cdd.lib"

!ENDIF 

# Begin Target

# Name "ta_libc - Win32 CDR Multithread DLL Release"
# Name "ta_libc - Win32 CMD Multithread Debug"
# Name "ta_libc - Win32 CSD Single Thread Debug"
# Name "ta_libc - Win32 CSR Single Thread Release"
# Name "ta_libc - Win32 CMR Multithread Release"
# Name "ta_libc - Win32 Profiling"
# Name "ta_libc - Win32 CDD Multithread DLL Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\..\..\..\include\ta_abstract.h
# End Source File
# Begin Source File

SOURCE=..\..\..\..\include\ta_common.h
# End Source File
# Begin Source File

SOURCE=..\..\..\..\include\ta_defs.h
# End Source File
# Begin Source File

SOURCE=..\..\..\..\include\ta_func.h
# End Source File
# Begin Source File

SOURCE=..\..\..\..\include\ta_libc.h
# End Source File
# End Group
# Begin Source File

SOURCE=..\..\..\..\..\CHANGELOG.TXT
# End Source File
# Begin Source File

SOURCE=..\..\..\..\..\HISTORY.TXT
# End Source File
# End Target
# End Project
