#!
#! This is a tmake template for building MSVC++ project files (.dsp)
#!
#${
    if ( Config("qt") ) {
	if ( !(Project("DEFINES") =~ /QT_NODLL/) &&
	     ((Project("DEFINES") =~ /QT_(?:MAKE)?DLL/) ||
	      ($ENV{"QT_DLL"} && !$ENV{"QT_NODLL"})) ) {
	    Project('TMAKE_QT_DLL = 1');
	    if ( (Project("TARGET") eq "qt" || Project("TARGET") eq "qt-mt" ) && Project("TMAKE_LIB_FLAG") ) {
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
    if ( Config("qt") || Config("opengl") ) {
	Project('CONFIG += windows' );
    }
    if ( Config("qt") ) {
	Project('CONFIG *= moc');
	Project('DEFINES *= UNICODE' );
	AddIncludePath(Project("TMAKE_INCDIR_QT"));
	Project('TMAKE_LIBS *= imm32.lib wsock32.lib winmm.lib');
	if ( Config("opengl") ) {
	    Project('TMAKE_LIBS *= $$TMAKE_LIBS_QT_OPENGL');
	}
	if ( (Project("TARGET") eq "qt" || Project("TARGET") eq "qt-mt") && Project("TMAKE_LIB_FLAG") ) {
	    if ( Project("TMAKE_QT_DLL") ) {
		Project('DEFINES *= QT_MAKEDLL');
		Project('MSVCDSP_DLLBASE = /base:"0x39D00000"');
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
    Project( 'DEFINES *= QT_THREAD_SUPPORT' );
    if ( Config("debug") ) {
	$project{"MSVCDSP_MTDEF"} = "-MDd";
    } else {
	$project{"MSVCDSP_MTDEF"} = "-MD";
    }
    if ( Config("dll") ) {
	if ( Project("TMAKE_LIB_FLAG") ) {
	    my $ver = Project("VERSION");
	    $ver =~ s/\.//g;
	    $project{"TARGET_EXT"} = "${ver}.dll";
	} else {
	    $project{"TARGET_EXT"} = ".dll";
	}
    } else {
	if ( Project("TMAKE_APP_FLAG") ) {
	    $project{"TARGET_EXT"} = ".exe";
	} else {
	    $project{"TARGET_EXT"} = ".lib";
	}
    }
    $project{"TARGET"} .= $project{"TARGET_EXT"};
    if ( Config("moc") ) {
	$moc_aware = 1;
    }
    Project('TMAKE_LIBS += $$LIBS');
    Project('TMAKE_FILETAGS = HEADERS SOURCES DEF_FILE RC_FILE TARGET TMAKE_LIBS DESTDIR DLLDESTDIR $$FILETAGS');
    foreach ( split(/\s/,Project("TMAKE_FILETAGS")) ) {
	$project{$_} =~ s-[/\\]+-\\-g;
    }
    StdInit();
    if ( check_unix() ) {
	$is_msvc5 = 0;
    } else {
	tmake_use_win32_registry();
	$HKEY_CURRENT_USER->Open("Software\\Microsoft\\DevStudio\\5.0",$is_msvc5);
    }
    if ( $is_msvc5 ) {
	$project{"MSVCDSP_VER"} = "5.00";
	$project{"MSVCDSP_DEBUG_OPT"} = "/Zi";
    } else {
	$project{"MSVCDSP_VER"} = "6.00";
	$project{"MSVCDSP_DEBUG_OPT"} = "/GZ /ZI";
    }
    $project{"MSVCDSP_PROJECT"} = $project{"OUTFILE"};
    $project{"MSVCDSP_PROJECT"} =~ s/\.[a-zA-Z0-9_]*$//;

    if ( Project("TMAKE_APP_FLAG") ) {
	$project{"MSVCDSP_TEMPLATE"} = "win32app.dsp";
	if ( Config("console") ) {
	    $project{"MSVCDSP_CONSOLE"} = "Console";
	    $project{"MSVCDSP_WINCONDEF"} = "_CONSOLE";
	    $project{"MSVCDSP_DSPTYPE"} = "0x0103";
	    $project{"MSVCDSP_SUBSYSTEM"} = "console";
	} else {
	    $project{"MSVCDSP_CONSOLE"} = "";
	    $project{"MSVCDSP_WINCONDEF"} = "_WINDOWS";
	    $project{"MSVCDSP_DSPTYPE"} = "0x0101";
	    $project{"MSVCDSP_SUBSYSTEM"} = "windows";
	}
    } else {
	if ( Config("dll") ) {
	    $project{"MSVCDSP_TEMPLATE"} = "win32dll.dsp";
	} else {
	    $project{"MSVCDSP_TEMPLATE"} = "win32lib.dsp";
	}
    }
    $project{"MSVCDSP_LIBS"} = $project{"TMAKE_LIBS"};
    ExpandGlue("DEFINES",'/D "','" /D "','"');
    $project{"MSVCDSP_DEFINES"} = $text; $text = "";
    ExpandPath("INCPATH",'/I ',' /I ','');
    $project{"MSVCDSP_INCPATH"} = $text; $text = "";
    if ( Config("qt") ) {
	$project{"MSVCDSP_RELDEFS"} = '/D "NO_DEBUG"';
    } else {
	$project{"MSVCDSP_RELDEFS"} = '';
    }
    if ( defined($project{"DESTDIR"}) ) {
	$project{"TARGET"} = $project{"DESTDIR"} . "\\" . $project{"TARGET"};
	$project{"TARGET"} =~ s/\\+/\\/g;
	$project{"MSVCDSP_TARGET"} = '/out:"' . $project{"TARGET"} . '"';
	if ( Config("dll") ) {
	    my $t = $project{"TARGET"};
	    $t =~ s/\.dll/.lib/;
	    $project{"MSVCDSP_TARGET"} .= " /implib:\"$t\"";
	}
    }
    if ( Config("dll") && Project("DLLDESTDIR") ) {
	$project{"MSVCDSP_COPY_DLL"} =
	  "# Begin Special Build Tool\n" .
	  "TargetPath=" . $project{"TARGET"} . "\n" .
	  "SOURCE=\$(InputPath)\n" .
	  "PostBuild_Desc=Copy DLL to " . $project{"DLLDESTDIR"} . "\n" .
	  "PostBuild_Cmds=copy \$(TargetPath) \"" . $project{"DLLDESTDIR"} . "\"\n" .
	  "# End Special Build Tool";
    }
    if ( Project("DSP_TEMPLATE") ) {
	$dspfile = Project("DSP_TEMPLATE");
    } else {
	$dspfile = Project("MSVCDSP_TEMPLATE");
    }
    $dsppath= &fix_path( &find_template($dspfile) );
    if ( !open(DSP,$dsppath) ) {
	&tmake_error("Cannot open dsp template $dspfile at $dsppath");
    }
    if ( Config("moc") ) {
	$project{"SOURCES"} .= " " . $project{"SRCMOC"};
    }
    if ( $project{"SOURCES"} || $project{"RC_FILE"} ) {
	$project{"SOURCES"} .= " " . $project{"RC_FILE"};
	@files = split(/\s+/,$project{"SOURCES"}); $text = "";
	foreach ( @files ) {
	    $file = $_;
	    $text .= "# Begin Source File\n\nSOURCE=.\\$file\n";
	    if ( Config("moc") && ($file =~ /\.moc$/) ) {
		$build = "\n\n# Begin Custom Build - Moc'ing $moc_input{$file}...\n" .
			 "InputPath=.\\$file\n\n" .
			 '"' . $file .
			     '" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"' . "\n" .
			 "\t%QTDIR%\\bin\\moc.exe " . $moc_input{$file} . " -o $file\n\n" .
			 "# End Custom Build\n\n";
		$base = $file;
		$base =~ s/\..*//;
		$base =~ tr/a-z/A-Z/;
		$base =~ s/[^A-Z]/_/g;
		$text .= "USERDEP_$base=" . '"' . $moc_input{$file} . '"' .
			 "\n\n" . '!IF  "$(CFG)" == "' .
			 $project{"MSVCDSP_PROJECT"} . ' - Win32 Release"' .
			 $build . '!ELSEIF  "$(CFG)" == "' .
			 $project{"MSVCDSP_PROJECT"} . ' - Win32 Debug"' .
			 $build . "!ENDIF \n\n";
	    }
	    $text .= "# End Source File\n";
	}
	$project{"MSVCDSP_SOURCES"} = $text; $text = "";
    }
    if ( $project{"HEADERS"} ) {
	@files = split(/\s+/,$project{"HEADERS"}); $text = "";
	foreach ( @files ) {
	    $file = $_;
	    $text .= "# Begin Source File\n\nSOURCE=.\\$file\n";
	    if ( Config("moc") && $moc_output{$file} ) {
		$build = "\n\n# Begin Custom Build - Moc'ing $file...\n" .
			 "InputPath=.\\$file\n\n" .
			 '"' . $moc_output{$file} .
			     '" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"' . "\n" .
			 "\t%QTDIR%\\bin\\moc.exe $file -o " . $moc_output{$file} . "\n\n" .
			 "# End Custom Build\n\n";
		$text .= "\n" . '!IF  "$(CFG)" == "' .
			 $project{"MSVCDSP_PROJECT"} . ' - Win32 Release"' .
			 $build . '!ELSEIF  "$(CFG)" == "' .
			 $project{"MSVCDSP_PROJECT"} . ' - Win32 Debug"' .
			 $build . "!ENDIF \n\n";
	    }
	    $text .= "# End Source File\n";
	}
	$project{"MSVCDSP_HEADERS"} = $text; $text = "";
    }
    if ($project{"INTERFACES"} ) {
	$uicpath = Expand("TMAKE_UIC");
	$uicpath =~ s/[.]exe//g;
	$uicpath .= " ";
	@files = split(/\s+/,$project{"INTERFACES"}); $text = ""; $headtext = ""; $sourcetext = "";
	foreach ( @files ) {
	    $file = $_;
	    $filename = $file;
	    $filename =~ s/[.]ui//g;
	    $text .= "# Begin Source File\n\nSOURCE=.\\$file\n";

	    $build = "\n\n# Begin Custom Build - Uic'ing $file...\n" .
			"InputPath=.\\$file\n\n" .
			"BuildCmds= " . $uicpath . $file . 
			" -o " . $filename . ".h\\\n" .
			"\t" . $uicpath . $file .
			" -i " . $filename . ".h -o " . $filename . ".cpp\\\n" .
			"\t%QTDIR%\\bin\\moc " . $filename . ".h -o " . $project{"MOC_DIR"} . "moc_" . $filename . ".cpp \\\n\n" .
			'"' . $filename . '.h" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"' . "\n" .
			"\t\$(BuildCmds)\n\n" .
			'"' . $filename . '.cpp" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"' . "\n" .
			"\t\$(BuildCmds)\n\n" .
			'"' . $project{"MOC_DIR"} . 'moc_' . $filename . '.cpp" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"' . "\n" .
			"\t\$(BuildCmds)\n\n" .
		"# End Custom Build\n\n";

	    $text .= '!IF  "$(CFG)" == "' . 
		    $project{"MSVCDSP_PROJECT"} . ' - Win32 Release"' . $build . 
			'!ELSEIF  "$(CFG)" == "' .
		    $project{"MSVCDSP_PROJECT"} . ' - Win32 Debug"' . $build .
		"!ENDIF \n\n";

	    $text .= "# End Source File\n";

		$sourcetext .= "# Begin Source File\n\nSOURCE=.\\" . $filename . ".cpp\n# End Source File\n";
		$headtext .= "# Begin Source File\n\nSOURCE=.\\" . $filename . ".h\n# End Source File\n";

	}
	$project{"MSVCDSP_INTERFACES"} = $text; $text = "";
	$project{"MSVCDSP_INTERFACESOURCES"} = $sourcetext; $sourcetext = "";
	$project{"MSVCDSP_INTERFACEHEADERS"} = $headtext; $headtext = "";
    }
    while ( <DSP> ) {
	$line = $_;
	while ( $line =~ s/((\s*)\$\$([a-zA-Z0-9_]+))/__MSVCDSP_SUBST__/ ) {
	    if ( defined($project{$3}) && ($project{$3} ne "")) {
		$subst = $project{$3};
		$space = $2;
		$line =~ s/__MSVCDSP_SUBST__/${space}${subst}/;
		if ( $line =~ /^\s*$/ ) {
		    $line = "";
		}
	    } else {
		$line =~ s/__MSVCDSP_SUBST__//;
	    }
	}
	$text .= $line;
    }
    close(DSP);
#$}
