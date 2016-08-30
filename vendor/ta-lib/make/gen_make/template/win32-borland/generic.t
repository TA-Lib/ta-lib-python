#!
#! This is a tmake template for building Win32 applications or libraries.
#!
#${
    Project('CONFIG += qt') if Config("qt_dll");
    if ( !Project("INTERFACE_DECL_PATH") ) {
	Project('INTERFACE_DECL_PATH = .' );
    }
    if ( Config("qt") ) {
	if ( !(Project("DEFINES") =~ /QT_NODLL/) &&
	     ((Project("DEFINES") =~ /QT_(?:MAKE)?DLL/) || Config("qt_dll") ||
	      ($ENV{"QT_DLL"} && !$ENV{"QT_NODLL"})) ) {
	    Project('TMAKE_QT_DLL = 1');
	    if ( (Project("TARGET") eq "qt") && Project("TMAKE_LIB_FLAG") ) {
		Project('CONFIG += dll');
	    }
	}
    }
    if ( Config("dll") || Project("TMAKE_APP_FLAG") ) {
	Project('CONFIG -= staticlib');
	Project('TMAKE_APP_OR_DLL = 1');
    } else {
	Project('CONFIG += staticlib');
    }
    if ( Config("warn_off") ) {
	Project('TMAKE_CFLAGS += $$TMAKE_CFLAGS_WARN_OFF');
	Project('TMAKE_CXXFLAGS += $$TMAKE_CXXFLAGS_WARN_OFF');
    } elsif ( Config("warn_on") ) {
	Project('TMAKE_CFLAGS += $$TMAKE_CFLAGS_WARN_ON');
	Project('TMAKE_CXXFLAGS += $$TMAKE_CXXFLAGS_WARN_ON');
    }
    if ( Config("thread") ) {
        Project('DEFINES += QT_THREAD_SUPPORT');

    }
    if ( Config("debug") ) {
        if ( Config("thread") ) {
	    if ( Config("dll") ) {
	        Project('TMAKE_CFLAGS += $$TMAKE_CFLAGS_MT_DLLDBG');
	        Project('TMAKE_CXXFLAGS += $$TMAKE_CXXFLAGS_MT_DLLDBG');
 	    } else {
		Project('TMAKE_CFLAGS += $$TMAKE_CFLAGS_MT_DBG');
		Project('TMAKE_CXXFLAGS += $$TMAKE_CXXFLAGS_MT_DBG');
	    }
        } else {
	    Project('TMAKE_CFLAGS += $$TMAKE_CFLAGS_DEBUG');
	    Project('TMAKE_CXXFLAGS += $$TMAKE_CXXFLAGS_DEBUG');
	}
	Project('TMAKE_LFLAGS += $$TMAKE_LFLAGS_DEBUG');
    } elsif ( Config("release") ) {
	if ( Config("thread") ) {
	    if ( Config("dll") ) {
		Project('TMAKE_CFLAGS += $$TMAKE_CFLAGS_MT_DLL');
		Project('TMAKE_CXXFLAGS += $$TMAKE_CXXFLAGS_MT_DLL');
	    } else {
		Project('TMAKE_CFLAGS += $$TMAKE_CFLAGS_MT');
		Project('TMAKE_CXXFLAGS += $$TMAKE_CXXFLAGS_MT');
	    }
	}
	Project('TMAKE_CFLAGS += $$TMAKE_CFLAGS_RELEASE');
	Project('TMAKE_CXXFLAGS += $$TMAKE_CXXFLAGS_RELEASE');
	Project('TMAKE_LFLAGS += $$TMAKE_LFLAGS_RELEASE');
    }

    if ( Project("TMAKE_INCDIR") ) {
	AddIncludePath(Project("TMAKE_INCDIR"));
    }
    if ( Config("qt") || Config("opengl") ) {
	Project('CONFIG += windows' );
    }
    if ( Config("qt") ) {
	Project('CONFIG *= moc');
	AddIncludePath(Project("TMAKE_INCDIR_QT"));
	if ( !Config("debug") ) {
	    Project('DEFINES += NO_DEBUG');
	}
	if ( (Project("TARGET") eq "qt") && Project("TMAKE_LIB_FLAG") ) {
	    if ( Project("TMAKE_QT_DLL") ) {
		Project('DEFINES *= QT_MAKEDLL');
		Project('TMAKE_LFLAGS += $$TMAKE_LFLAGS_QT_DLL');
	    }
	} else {
	    Project('TMAKE_LIBS *= $$TMAKE_LIBS_QT');
	    if ( Project("TMAKE_QT_DLL") ) {
		my $qtver =FindHighestLibVersion($ENV{"QTDIR"} . "/lib", "qt");
		Project("TMAKE_LIBS /= s/qt.lib/qt${qtver}.lib/");
		if ( !Config("dll") ) {
		    Project('TMAKE_LIBS *= $$TMAKE_LIBS_QT_DLL');
		}
	    }
	}
    }
    if ( Config("opengl") ) {
	Project('TMAKE_LIBS *= $$TMAKE_LIBS_OPENGL');
    }
    if ( Config("dll") ) {
	Project('TMAKE_LFLAGS_CONSOLE_ANY = $$TMAKE_LFLAGS_CONSOLE_DLL');
	Project('TMAKE_LFLAGS_WINDOWS_ANY = $$TMAKE_LFLAGS_WINDOWS_DLL');
	if ( Project("TMAKE_LIB_FLAG") ) {
	    my $ver = Project("VERSION");
	    $ver =~ s/\.//g;
	    $project{"TARGET_EXT"} = "${ver}.dll";
	} else {
	    $project{"TARGET_EXT"} = ".dll";
	}
    } else {
	Project('TMAKE_LFLAGS_CONSOLE_ANY = $$TMAKE_LFLAGS_CONSOLE');
	Project('TMAKE_LFLAGS_WINDOWS_ANY = $$TMAKE_LFLAGS_WINDOWS');
	if ( Project("TMAKE_APP_FLAG") ) {
	    $project{"TARGET_EXT"} = ".exe";
	} else {
	    $project{"TARGET_EXT"} = ".lib";
	}
    }
    if ( Config("windows") ) {
	if ( Config("console") ) {
	    Project('TMAKE_LFLAGS *= $$TMAKE_LFLAGS_CONSOLE_ANY');
	    Project('TMAKE_LIBS   *= $$TMAKE_LIBS_CONSOLE');
	} else {
	    Project('TMAKE_LFLAGS *= $$TMAKE_LFLAGS_WINDOWS_ANY');
	}
	Project('TMAKE_LIBS   *= $$TMAKE_LIBS_WINDOWS');
    } else {
	Project('TMAKE_LFLAGS *= $$TMAKE_LFLAGS_CONSOLE_ANY');
	Project('TMAKE_LIBS   *= $$TMAKE_LIBS_CONSOLE');
    }
    if ( Config("thread") ) {
        Project('TMAKE_LIBS   *= $$TMAKE_LIBS_RTMT');
    } else {
        Project('TMAKE_LIBS   *= $$TMAKE_LIBS_RT');
    }
    if ( Config("moc") ) {
	$moc_aware = 1;
    }
    Project('TMAKE_LIBS += $$LIBS');
    Project('TMAKE_FILETAGS = HEADERS SOURCES DEF_FILE RC_FILE TARGET TMAKE_LIBS DESTDIR DLLDESTDIR $$FILETAGS');
    foreach ( split(/\s/,Project("TMAKE_FILETAGS")) ) {
	$project{$_} =~ s-[/\\]+-\\-g;
    }
    if ( Project("RC_FILE") ) {
	if ( Project("RES_FILE") ) {
	    tmake_error("Both .rc and .res file specified.\n" .
			"Please specify one of them, not both.");
	}
	$project{"RES_FILE"} = $project{"RC_FILE"};
	$project{"RES_FILE"} =~ s/\.rc$/.res/i;
	Project('TARGETDEPS += $$RES_FILE');
    }
    StdInit();
    if ( Project("VERSION") ) {
	$project{"VER_MAJ"} = $project{"VERSION"};
	$project{"VER_MAJ"} =~ s/\.\d+$//;
	$project{"VER_MIN"} = $project{"VERSION"};
	$project{"VER_MIN"} =~ s/^\d+\.//;
    }
    Project('TMAKE_CLEAN += $$TARGET.tds');
#$}
#!
# Makefile for building #$ Expand("TARGET")
# Generated by tmake;
#     Project: #$ Expand("PROJECT");
#    Template: #$ Expand("TEMPLATE");
#############################################################################

!if !$d(BCB)
BCB = $(MAKEDIR)\..
!endif

####### Compiler, tools and options

CC	=	#$ Expand("TMAKE_CC");
CXX	=	#$ Expand("TMAKE_CXX");
CFLAGS	=	#$ Expand("TMAKE_CFLAGS"); ExpandGlue("DEFINES","-D"," -D","");
CXXFLAGS=	#$ Expand("TMAKE_CXXFLAGS"); ExpandGlue("DEFINES","-D"," -D","");
INCPATH	=	#$ ExpandPath("INCPATH",'-I',' -I','');
#$ !Project("TMAKE_APP_OR_DLL") && DisableOutput();
LINK	=	#$ Expand("TMAKE_LINK");
LFLAGS	=	#$ Expand("TMAKE_LFLAGS");
LIBS	=	#$ Expand("TMAKE_LIBS");
#$ !Project("TMAKE_APP_OR_DLL") && EnableOutput();
#$ Project("TMAKE_APP_OR_DLL") && DisableOutput();
LIB	=	#$ Expand("TMAKE_LIB");
#$ Project("TMAKE_APP_OR_DLL") && EnableOutput();
MOC	=	#$ Expand("TMAKE_MOC");
UIC	=	#$ Expand("TMAKE_UIC");

ZIP	=	#$ Expand("TMAKE_ZIP");
DEF_FILE =	#$ ExpandList("DEF_FILE");
RES_FILE =	#$ ExpandList("RES_FILE");

####### Files

HEADERS =	#$ ExpandList("HEADERS");
SOURCES =	#$ ExpandList("SOURCES");
OBJECTS =	#$ ExpandList("OBJECTS");
INTERFACES =	#$ ExpandList("INTERFACES");
UICDECLS =	#$ ExpandList("UICDECLS");
UICIMPLS =	#$ ExpandList("UICIMPLS");
SRCMOC	=	#$ ExpandList("SRCMOC");
OBJMOC	=	#$ ExpandList("OBJMOC");
DIST	=	#$ ExpandList("DISTFILES");
TARGET	=	#$ ExpandGlue("TARGET",$project{"DESTDIR"},"",$project{"TARGET_EXT"});
INTERFACE_DECL_PATH = #$ Expand("INTERFACE_DECL_PATH");

####### Implicit rules

.SUFFIXES: .cpp .cxx .cc .c

.cpp.obj:
	#$ Expand("TMAKE_RUN_CXX_IMP");

.cxx.obj:
	#$ Expand("TMAKE_RUN_CXX_IMP");

.cc.obj:
	#$ Expand("TMAKE_RUN_CXX_IMP");

.c.obj:
	#$ Expand("TMAKE_RUN_CC_IMP");

####### Build rules

all: #$ ExpandGlue("ALL_DEPS",""," "," "); $text .= '$(TARGET)';

$(TARGET): $(UICDECLS) $(OBJECTS) $(OBJMOC) #$ Expand("TARGETDEPS");
#$ Project("TMAKE_APP_OR_DLL") || DisableOutput();
	$(LINK) @&&|
	    $(LFLAGS) $(OBJECTS) $(OBJMOC),$(TARGET),,$(LIBS),$(DEF_FILE),$(RES_FILE)
#$ Project("TMAKE_APP_OR_DLL") || EnableOutput();
#$ Project("TMAKE_APP_OR_DLL") && DisableOutput();
	-del $(TARGET)
	$(LIB) $(TARGET) @&&|
#${
# $text = "+" . join(" \\\n+",split(/\s+/,$project{"OBJECTS"})) . " \\\n+"
#             . join(" \\\n+",split(/\s+/,$project{"OBJMOC"}));
#$}
#$ Project("TMAKE_APP_OR_DLL") && EnableOutput();
|
#$ (Config("dll") && Project("DLLDESTDIR")) || DisableOutput();
	-copy $(TARGET) #$ Expand("DLLDESTDIR");
#$ (Config("dll") && Project("DLLDESTDIR")) || EnableOutput();
#$ Project("RC_FILE") || DisableOutput();

#$ Substitute("\$\$RES_FILE: \$\$RC_FILE\n\t\$\$TMAKE_RC \$\$RC_FILE");
#$ Project("RC_FILE") || EnableOutput();

moc: $(SRCMOC)

#$ TmakeSelf();

dist:
	#$ Substitute('$(ZIP) $$PROJECT.zip $$PROJECT.pro $(SOURCES) $(HEADERS) $(DIST)');

clean:
	#$ ExpandGlue("OBJECTS","-del ","\n\t-del ","");
	#$ ExpandGlue("SRCMOC" ,"-del ","\n\t-del ","");
	#$ ExpandGlue("OBJMOC" ,"-del ","\n\t-del ","");
	-del $(TARGET)
	#$ ExpandGlue("TMAKE_CLEAN","-del ","\n\t-del ","");
	#$ ExpandGlue("CLEAN_FILES","-del ","\n\t-del ","");

####### Compile

#$ BuildObj(Project("OBJECTS"),Project("SOURCES"));
#$ BuildUicSrc(Project("INTERFACES"));
#$ BuildObj(Project("UICOBJECTS"), Project("UICIMPLS"));
#$ BuildMocObj(Project("OBJMOC"),Project("SRCMOC"));
#$ BuildMocSrc(Project("HEADERS"));
#$ BuildMocSrc(Project("SOURCES"));
#$ BuildMocSrc(Project("UICDECLS"));
