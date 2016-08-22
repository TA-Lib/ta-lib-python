# Microsoft Developer Studio Project File - Name="$$MSVCDSP_PROJECT" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version $$MSVCDSP_VER
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) $$MSVCDSP_CONSOLE Application" $$MSVCDSP_DSPTYPE

CFG=$$MSVCDSP_PROJECT - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "$$MSVCDSP_PROJECT.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "$$MSVCDSP_PROJECT.mak" CFG="$$MSVCDSP_PROJECT - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "$$MSVCDSP_PROJECT - Win32 Release" (based on "Win32 (x86) Application")
!MESSAGE "$$MSVCDSP_PROJECT - Win32 Debug" (based on "Win32 (x86) Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "$$MSVCDSP_PROJECT - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo $$MSVCDSP_MTDEF /W3 /O1 /D "WIN32" /D "NDEBUG" /D "$$MSVCDSP_WINCONDEF" /D "_MBCS" /FD /c
# ADD CPP /nologo $$MSVCDSP_MTDEF /W3 /O1 $$MSVCDSP_INCPATH /D "WIN32" /D "NDEBUG" /D "$$MSVCDSP_WINCONDEF" /D "_MBCS" $$MSVCDSP_DEFINES $$MSVCDSP_RELDEFS /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib imm32.lib winmm.lib wsock32.lib /nologo /subsystem:$$MSVCDSP_SUBSYSTEM /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib imm32.lib winmm.lib wsock32.lib $$MSVCDSP_LIBS /nologo /subsystem:$$MSVCDSP_SUBSYSTEM /machine:I386 $$MSVCDSP_TARGET

!ELSEIF  "$(CFG)" == "$$MSVCDSP_PROJECT - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo $$MSVCDSP_MTDEF /W3 /Gm $$MSVCDSP_DEBUG_OPT /Od /D "WIN32" /D "_DEBUG" /D "$$MSVCDSP_WINCONDEF" /D "_MBCS" /FD /c
# ADD CPP /nologo $$MSVCDSP_MTDEF /W3 /Gm $$MSVCDSP_DEBUG_OPT /Od $$MSVCDSP_INCPATH /D "WIN32" /D "_DEBUG" /D "$$MSVCDSP_WINCONDEF" /D "_MBCS" $$MSVCDSP_DEFINES /FD /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib imm32.lib winmm.lib wsock32.lib /nologo /subsystem:$$MSVCDSP_SUBSYSTEM /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib imm32.lib winmm.lib wsock32.lib $$MSVCDSP_LIBS /nologo /subsystem:$$MSVCDSP_SUBSYSTEM /debug /machine:I386 $$MSVCDSP_TARGET /nodefaultlib:"libc" /pdbtype:sept

!ENDIF 

# Begin Target

# Name "$$MSVCDSP_PROJECT - Win32 Release"
# Name "$$MSVCDSP_PROJECT - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
$$MSVCDSP_SOURCES
$$MSVCDSP_INTERFACESOURCES
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
$$MSVCDSP_HEADERS
$$MSVCDSP_INTERFACEHEADERS
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# Begin Group "Interfaces"
$$MSVCDSP_INTERFACES
# Prop Default_Filter "ui"
# End Group
# End Target
# End Project
