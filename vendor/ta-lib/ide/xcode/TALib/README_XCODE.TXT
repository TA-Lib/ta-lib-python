* Information for Xcode Developers on Mac OS X *
* by Drew McCormack                            *

What does this Xcode project build?
===================================
This Xcode project builds a Mac OS X framework of the TA-LIB functions.
The native Xcode build system is used, rather than the make utility used 
on other UNIX systems. The framework build is single-threaded.

Where does the framework get built?
===================================
Upon completion of a successful build, the framework TALib.framework
should appear in the 'build' directory inside the TALib.xcode project
directory, or in your default Xcode build directory.

Where should the framework be installed for use in an application?
==================================================================
The Xcode project is setup such that the TALib.framework file should be
installed in the 'Frameworks' directory inside your application bundle. (You
can copy the TALib.framework to that directory by adding a 'Copy Phase'
to your application's target. See Xcode documentation for details.) 

If you wish to change the install path of the framework, to /Library/Frameworks
for example, you can simply double click on the TALib target in Xcode, 
select the 'Build' tab, and change the 'Installation Directory' setting 
appropriately.

What frameworks/libraries does the TALib framework use?
=======================================================
The TALib framework requires libbz2 and libcurl. Both of these
libraries are preinstalled in Mac OS X 10.3 and 10.4. You can either use the 
versions installed with Mac OS X, or include your own versions. 

Is the TALib framework for Cocoa or Carbon applications?
========================================================
TALib is a C library, so it can be called from Cocoa or Carbon. There are 
no restrictions in this regard.
