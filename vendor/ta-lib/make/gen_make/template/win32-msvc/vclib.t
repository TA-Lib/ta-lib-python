#!
#! This is a tmake template for building Win32 library project files.
#!
#! Sets a flag to indicate that we want to build a library (either
#! a static library or a DLL) and then invoke the common vcgeneric.t
#! template.
#!
#! The win32lib.dsp file is used as a template for building static
#! libraries and win32dll.dsp is used as a template for building DLLs.
#! You may specify your own .dsp template by setting the project variable
#! DSP_TEMPLATE.
#!
#$ Project('TMAKE_LIB_FLAG = 1');
#$ IncludeTemplate("vcgeneric.t");
