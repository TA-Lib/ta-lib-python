#!/usr/bin/perl

# This perl script is the "mother of all" making
# all the operations for re-generating all the
# makefiles variant.

if( $^O eq "MSWin32" )
{
   $ENV{'TMAKEPATH'} = './template/win32-msvc';
   $MAKEPATH = './template/win32-msvc';
}
else
{
   $ENV{'TMAKEPATH'} = './template/linux-g++';
   $MAKEPATH = './template/linux-g++';
}

print "Generating ta_func.pro template...";
chdir "ta_func";
system( "perl make_pro.pl >ta_func.pro" );
chdir "..";
print "done.\n";

print "Generating ta_abstract.pro template...";
chdir "ta_abstract";
system( "perl make_pro.pl >ta_abstract.pro" );
chdir "..";
print "done.\n";

print "Generating ta_libc.pro template...";
chdir "ta_libc";
system( "perl make_pro.pl >ta_libc.pro" );
chdir "..";
print "done.\n";

system( "perl ./make_make.pl cdr $MAKEPATH \"template/*\" all" );
system( "perl ./make_make.pl cdd $MAKEPATH \"template/*\" all" );
system( "perl ./make_make.pl cmd $MAKEPATH \"template/*\" all" );
system( "perl ./make_make.pl cmr $MAKEPATH \"template/*\" all" );
system( "perl ./make_make.pl cmp $MAKEPATH \"template/*\" all" );
system( "perl ./make_make.pl csr $MAKEPATH \"template/*\" all" );
system( "perl ./make_make.pl csd $MAKEPATH \"template/*\" all" );
system( "perl ./make_make.pl csp $MAKEPATH \"template/*\" all" );
