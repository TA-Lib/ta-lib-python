#!
#! This is a tmake template for building Win32 application project files.
#!
#! Sets a flag to indicate that we want to build an application and then
#! invoke the common vcgeneric.t template.
#!
#! The win32app.dsp file is used as a template for building applications.
#! You may specify your own .dsp template by setting the project variable
#! DSP_TEMPLATE.
#!
#$ Project('TMAKE_APP_FLAG = 1');
#$ IncludeTemplate("vcgeneric.t");
