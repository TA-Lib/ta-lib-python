These directories contains the Makefiles
for building TA-Lib.

The first level of directory allows to identify
the kind of application environment in which the
static library is going to be used:

  CSR - Single thread
  CMR - Multi  thread
  CSD - Single thread with debug options
  CMD - Multi  thread with debug options
  CSP - Single thread with profiling options
  CMP - Multi  thread with profiling options
  CDR - DLL Release (Win32 only)
  CDD - DLL Debug (Win32 only)

Choose the most appropriate directory depending of
your platform/compiler and run "make" or "nmake"
from there.

See the API Documentation for more information
on how to build and link with TA-Lib.

The directory gen_make contains the perl scripts
automating the generation of all the makefiles.
