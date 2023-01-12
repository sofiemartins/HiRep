#!/usr/bin/env perl
use warnings;
use strict;
use File::Path qw(make_path);
use Cwd qw(abs_path);
use List::MoreUtils qw(uniq);
# use Data::Dumper;

my $disable_color = 0;
my $create_vscode_c_cpp_properties = 1;

#package NinjaBuild;

my $inifile = "MkFlags.ini";
my $test_wrapper_cmd = "test_wrapper.pl";

my %options = ();
my $with_gpu = 0; #this is also in the %options
my $with_mpi = 0; #this is also in the %options

my %targets = (); #global list of objects for which rules have been emitted. This is to avoid duplicate rules
my $absroot; #absolute path to root project directory

sub build_rules {
    my ($rootdir) = @_; 
    print <<'EOF';
# This file is used to build ninja itself.
# It is generated by build.pl with NinjaBuild.pl

ninja_required_version = 1.3

EOF
    %options = read_conf("$rootdir/Make/$inifile");
    print_options(%options);
    $absroot = abs_path($rootdir);
    print("root = $absroot\n");
    print("TESTWRAPPER = \$root/Make/Utils/$test_wrapper_cmd" . ($with_mpi ? " --mpi\n" : "\n"));
    print <<'EOF';
builddir = $root/build
makedir = $root/Make
incdir = $root/Include
coreincdir = $root/Include/Core
AR = ar

INCLUDE = $INCLUDE -I$incdir -I$coreincdir
CFLAGS = -std=c99 $MACRO $CFLAGS $INCLUDE
GPUFLAGS = $MACRO $GPUFLAGS $INCLUDE
LDFLAGS = -L$builddir $LDFLAGS

wr_head = $makedir/Utils/write_suN_headers.pl
wr_repr = $makedir/Utils/autosun/writeREPR
wr_gpugeo = $makedir/Utils/write_gpu_geometry.pl

rule cc
  command = $CC -MMD -MF $out.d $CFLAGS -c $in -o $out
  description = $setbg CC $setnorm $out
  depfile = $out.d
  deps = gcc

rule nvcc
  command = $NVCC -ccbin $CC -MMD -MF $out.d $GPUFLAGS --device-c $in -o $out
  description = $setbg NVCC $setnorm $out
  depfile = $out.d
  deps = gcc

rule ar
  command = rm -f $out && $AR crs $out $in
  description = $setbg AR $setnorm $out

rule link
  command = $LINK -o $out $in $libs -lm $LDFLAGS
  description = $setbg LINK $setnorm $out 

rule test
  command = $TESTWRAPPER -t $root/$in
  description = $setbg TEST $setnorm $out
  pool = console

rule suN_headers
  command = cd $coreincdir && $wr_head $NG $REPR $WQUAT $GAUGE_GROUP
  description = $setbg SUN HEADERS $setnorm $out

rule suN_repr
  command = $wr_repr $NG $in > $out 2>/dev/null
  description = $setbg SUN REPRESENTATION $setnorm $out

rule gpu_geometry
  command = cd $outdir && $wr_gpugeo $NG $REPR $WQUAT $GAUGE_GROUP
  description = $setbg GPU GOMETRY HEADERS $setnorm $out

# build writeREPR
wr_repr_build = $builddir/Make/Utils/autosun/
build $wr_repr_build/writeREPR.o: cc $makedir/Utils/autosun/main.cc
  CC = $CXX
  CFLAGS = -D_${REPR}_ -D_${GAUGE_GROUP}_ -D_PICA_STYLE_ -DNDEBUG -O3
build $wr_repr: link $wr_repr_build/writeREPR.o
  LINK = $CXX
  LDFLAGS =
build writeREPR: phony $wr_repr

# Autoheaders
build autoheaders: phony $coreincdir/suN.h $coreincdir/suN_types.h $coreincdir/suN_repr_func.h $root/Include/Geometry/gpu_geometry.h
build $coreincdir/suN.h $coreincdir/suN_types.h: suN_headers | $wr_head
build $coreincdir/suN_repr_func.h: suN_repr $coreincdir/TMPL/suN_repr_func.h.tmpl | $wr_repr
build $root/Include/Geometry/gpu_geometry.h: gpu_geometry | $wr_gpugeo
  outdir = $root/Include/Geometry/

# LibHR/Update
updatedir = $root/LibHR/Update/
rule approx_db
  command = cd $updatedir/remez && cat approx_* | ./mkappdata.pl > $out
  description = $setbg APPROX DB $setnorm $out

build remez_db: phony $updatedir/approx_data.db
build $updatedir/approx_data.db: approx_db

# LibHR/Utils
# CInfo
rule cinfo
  command = cd $outdir && $makedir/Utils/cinfo.sh $makedir $root $MACRO 
  description = $setbg CINFO $setnorm $out

utilsdir = $root/LibHR/Utils/
build $utilsdir/cinfo.c: cinfo | $makedir/Utils/cinfo.sh
  outdir = $utilsdir

# autogen files
build generated_files: phony autoheaders remez_db

# Ad hoc rules - these are not build by default
build $root/ModeNumber/approx_for_modenumber.o: cc $root/ModeNumber/approx_for_modenumber.c
build $root/ModeNumber/approx_for_modenumber: link $root/ModeNumber/approx_for_modenumber.o
  LDFLAGS = -lm -lgsl -lgslcblas
build ModeNumber/approx_for_modenumber: phony $root/ModeNumber/approx_for_modenumber

# Doxygen documentation
docdir = $root/Doc/
rule docs
  command = cd $docdir && doxygen doxygen/Doxyfile 2>/dev/null >/dev/null 
  description = $setbg DOXYGEN $setnorm Documentation
build docs: docs
build Doc: phony docs

# Regenerate build files if build script changes.
rule configure
  command = $root/Make/build.pl > $out
  generator = 1
EOF
    print "build $absroot/build/build.ninja: configure | $absroot/Make/build.pl \$makedir/NinjaBuild.pl \$makedir/$inifile\n\n";
    if ($create_vscode_c_cpp_properties) { print_c_cpp_properties($absroot); }
}

# check if a string contains a substring
sub contains_substring {
    my ($string, $substring) = @_;
    if (index($string, $substring) != -1) {
        return 1;
    }
    return 0;
}
# check if an array of string contains a string
sub contains {
    my ($array_ref, $string) = @_;
    foreach (@{$array_ref}) {
        if ($_ eq $string) { return 1;}
    }
    return 0;
}

sub obj_rules {
    my ($name, @c_sources) = @_;
    my @c_objs = ();
    foreach ( @c_sources ) {
        my $obj = $_;
        $obj=~s/\.cu?$/\.o/;
        my $cc_rule = "cc";
        if (/\.cu$/) {
            $cc_rule = "nvcc";
            if(!$with_gpu) { next; } #skip cuda files if not with_gpu
        } 
        my $absobj = abs_path($obj);
        $absobj =~ s/^$absroot\///;
        $obj = "\$builddir/$absobj";
        if (not exists $targets{$obj}) {
            my $abssrc = abs_path("$_");
            print "build $obj: $cc_rule $abssrc || generated_files\n";
            $targets{$obj} = "";
        }
        # $obj .= " ";
        push(@c_objs, $obj);
    }
    print "build $name: phony @c_objs\n";
    return @c_objs;
}

sub lib_rules {
    my ($lib_name, @c_objs) = @_;
    my @unique_objs = uniq(sort(@c_objs)); #ensure objects are unique and sorted to ensure identical commans line
    print "build \$builddir/$lib_name: ar @unique_objs\n";
    print "build $lib_name: phony \$builddir/$lib_name\n";
}

sub exe_rules {
    my ($exe_name, $c_objs_ref, $libs_ref) = @_;
    # for dependencies we need absolute path to the buildir
    my @libs = ();
    foreach (@$libs_ref) { push(@libs,"\$builddir/$_");}
    print "build $exe_name: link @$c_objs_ref | @libs\n";
    # for the linker we need to transform libhr.a -> -lhr
    @libs = ();
    foreach (@$libs_ref) {
        $_ =~ /lib(.*)\..*/;
        push(@libs,"-l$1");
    }
    print "  libs = @libs\n";
}

sub delete_element {
    my ( $array_ref, $elem ) = @_;
    my @arr = @$array_ref;
    my $idx = 0;
    my $len = @arr;
    while (($idx<$len) and (index($$array_ref[$idx], $elem) == -1)) {
        $idx++;
    }
    splice(@$array_ref, $idx, 1);
}

sub exclude_files {
    my ( $array_ref, $exclude_ref ) = @_;
    foreach ( @$exclude_ref ) {
        delete_element($array_ref, $_);
    }
}

sub add_exes {
    my ($topdir, $exes_ref, $libs_ref) = @_;
    my @target_exes = ();
    while ( my ($k,$v) = each %$exes_ref ) {
        my @c_sources = map { "$topdir/$_" } @$v;
        # skip if NOCOMPILE is true
        if (no_compile($c_sources[0])) { next; }
        my $exe_name = "$topdir/$k";
        my @c_objs = obj_rules("${exe_name}_obj", @c_sources);
        exe_rules($exe_name, \@c_objs, $libs_ref);
        push(@target_exes,$exe_name);
    }
    print "build $topdir: phony @target_exes\n";
    return @target_exes;
    # print "default $topdir\n";
}


sub add_tests {
    my ($topdir, $tests_ref) = @_;
    my $alias = "build ${topdir}_tests: phony";
    foreach (@$tests_ref) {
        print("build ${_}_test: test $_\n");
        $alias .= " ${_}_test";
    }
    print("$alias\n");
}

#
# Parse compilation flags 
#
sub read_conf {
    my ($filename) = @_;
    open my $fh, '<', $filename or die "Can not open '$filename' $!";

    # parse INI file
    my %options;
    while (my $line = <$fh>) {
        if ($line =~ /^\s*$/) { next; }  # skip empty lines
        $line =~ s/#.*$//;  # strip comments
        
        # handle assignments = or +=
        if ($line =~ /^([^=]+?)\s*\+?=\s*(.*?)\s*$/) {
            my ($field, $value) = ($1, $2);
            if (not exists $options{$field}) { $options{$field} = []; }
            if ($value) { push(@{$options{$field}},"$value"); }
            next; 
        }
    }

    #TODO: we should do some input validation

    # set a default C++ compiler if none is given
    if (not exists $options{'CXX'}) {$options{'CXX'} = [ "g++" ] ; }
    
    # set WQUAT option
    if (contains($options{'MACRO'},"WITH_QUATERNIONS")) {
        $options{'WQUAT'} = [ "1" ] ; 
    } else {
        $options{'WQUAT'} = [ "0" ] ;
    }

    # handle WITH_MPI compiler 
    if (contains($options{'MACRO'},"WITH_MPI")) {
        $options{'CC'} = $options{'MPICC'};
        $with_mpi = 1;
    }

    #set linker 
    $options{'LINK'} = [ $options{'CC'}[0] ] ;

    # handle WITH_GPU
    if (contains($options{'MACRO'},"WITH_GPU")) {
        if (not exists $options{'NVCC'}) {
            die("'WITH_GPU' is set but no 'NVCC' compiler given!\n");
        }
        $with_gpu = 1;

        # find CUDA install dir
        # ideally we would ask the NVCC compiler, but it does not seems to be supported
        my $nvcc = $options{'NVCC'}[0];
        my $nvcc_path = `echo 'command -v $nvcc' | sh`;
        $nvcc_path =~ m{(.*?)/bin/$nvcc} or die("Cannot locate NVCC compiler [$nvcc]!\n");
        my $cuda_path = $1;
        # print ("CUDA= $cuda_path\n");
        # add standard CUDA include and lib dirs
        push(@{$options{'INCLUDE'}},"-I$cuda_path/include/");
        push(@{$options{'LDFLAGS'}},"-lcuda");

        #set linker 
        unshift @{$options{'LINK'}}, "$nvcc --forward-unknown-to-host-compiler -ccbin"; 
    }

    # add standard definitions to MACRO
    push(@{$options{'MACRO'}},"NG=${$options{'NG'}}[0]");
    push(@{$options{'MACRO'}},"${$options{'GAUGE_GROUP'}}[0]");
    push(@{$options{'MACRO'}},"${$options{'REPR'}}[0]");
    push(@{$options{'MACRO'}},"REPR_NAME=\\\"${$options{'REPR'}}[0]\\\"");

    #print Dumper(\%options);
    return %options;
}

sub print_options {
    my (%options) = @_ ;

    if (!$disable_color) {
        push(@{$options{'CFLAGS'}},"-fdiagnostics-color=always");
        # push(@{$options{'GPUFLAGS'}},"-Xcompiler '-fdiagnostics-color=always'");
        print "# Color Options\n\n";
        print "setbg = \e[07;1;31m\n";
        print "setnorm = \e[0m\n";
        print "\n";
    }

    print "# Compilation Options\n\n";
    while ( my ($k,$v) = each %options ) {
        if ($k eq "MACRO") {
            my @o = map { "-D" . $_ } @$v;
            print "$k = @o\n";
        } else {
            print "$k = @$v\n";
        }
    }
    print "\n";
}

#
# PARSE NOCOMPILE OPTIONS
#
sub no_compile {
    my $max_lines = 20; #max number of lines we read to process NOCOMPILE options
    my ($source) = @_;
    my @options = ();
    open my $FH,'<',$source or die "[$source]: $!\n";
    while (<$FH>) {
        if (/^\s*\*\s* NOCOMPILE\s*=\s*(.*?)\s*$/) { push(@options, $1); }
        last if $. >= $max_lines;
    }
    close $FH;
    # if no NOCOMPILE options are given exit
    if (!@options) { return 0; }

    # parse list of defined MACROS
    my @defined = map { local $_ = $_; s/=/==/g; s/=/_/g; s/[\"\\]//g; $_ }  @{$options{"MACRO"}};
    # create a dictionary where each defined macro is true = 1
    my %dictionary; foreach ( @defined ) { $dictionary{$_}=1; }
    # build logical expression and parse it
    my $nocomp_expr = "0";
    foreach( @options ) { $nocomp_expr .= "||($_)"; } # NOCOMPILE options on different lines are joined via logical OR
    #evaluate result
    my $result = parse_logical_expr($nocomp_expr, \%dictionary);
    # print "NOCOMP=$result\n";
}

my $parser_vars; # variable dictionary hash ref
sub parse_err { die s/\G/<* @_ *>/r, "\n" }
sub closeparen { /\G\)/gc ? shift : parse_err "missing )"; }

sub parse_expr {
    my $p = shift; # precedence
    my $answer = undef;
    /\G(?:\s+|#.*)+/gc; # skip whitespace or comments

    my $negate = 0;  while(/\G\!/gc) { $negate ^= 1; }
    if (/\G(\w+)\s*/gc) { 
        # print "NAME=$1 value=$parser_vars->{$1} prec=$p\n";
        $answer=(exists $parser_vars->{$1}) ? $parser_vars->{$1} : 0; 
    } elsif (/\G\(/gc) {
        $answer=closeparen(parse_expr(0));
    } else { parse_err "syntax error"; }

    /\G(?:\s+|#.*)+/gc; # skip whitespace or comments
    if ($p <= 3 && $negate) { $answer ^= 1; }
    while ($p <= 2 && /\G\&\&/gc) { $answer &= parse_expr(3); }
    while ($p <= 1 && /\G\|\|/gc) { $answer |= parse_expr(2); } # lowest precedence
    return $answer; 
}

sub parse_logical_expr { # takes expression string and ref to dictionary hash
  (local $_, $parser_vars) = @_;
  # we transform == into __ so that expressions like NG==3 are matched 
  # by an indetifier like NG__3
  s/=/_/g;
  my $result = parse_expr(0);
  /\G\z/gc or parse_err "incomplete parse";
  return $result;
}

# sub test_parser {
#     my @tests = grep /\S/, split /\n/, <<END;
# A || B
# A||(B)
# (A)||B
# (((A)||(B)))
# A && B
# (A && B) || C
# A && B || C
# A && (B || C)
# (A && (B || C))
# !A
# !(A)
# !(!A)
# !!A
# A || !B
# !(A || !B)
# !A || !B
# A || B && C
# A || (B && C)
# (A || B) && C
# A && B && C
# (A && B) && C
# A && (B && C)
# A || B || C
# (A || B) || C
# A || (B || C)
# A && !B && C
# A && B && C && D
# A||(!B||!(C&&D))
# !(A||(!B||!(C&&D)))
# END

#     for ( @tests ) {  # for each test case
#         print "\nparsing: $_\n";
#         for my $input (glob join '', map "$_:\{0,1\}", # for each combination
#             sort s/.*=|#.*//gr =~ /\w+/g)                # of input values
#         {
#             print "[$input] ";
#             my %dictionary = $input =~ /(\w+):(0|1)/g;
#             my $result = parse( $_, \%dictionary );
#             print "= $result\n";
#         }
#     }
# }

#
# CONFIGURATION FOR VSCODE
#
sub print_c_cpp_properties {
    my ($rootdir) = @_;
    my $optfile = "$rootdir/.vscode/c_cpp_properties.json";
    my $compiler = `echo 'command -v ${$options{'CC'}}[0]' | sh`;
    chomp($compiler);
    my $compilermode = "gcc-x64";
    
    #make sure path exists
    make_path("$rootdir/.vscode/");

    open(FH, '>', $optfile) or die $!;
    print FH <<EOF;
{
    "configurations": [
        {
            "name": "HiRep",
            "defines": [
EOF
    foreach (@{$options{'MACRO'}}) {
        print FH "                \"$_\"";
        if (not ($_ eq ${$options{'MACRO'}}[-1])) {
            print FH ",";
        }
        print FH "\n";
    }
    my $includelist="";
    foreach(@{$options{"INCLUDE"}}) {
        /-I(.*)/ or die("Wrong format for INCLUDE!\n");
        $includelist .= ",\"$1\"";
    }
    print FH <<EOF;
            ],
            "compilerPath": "$compiler",
            "intelliSenseMode": "$compilermode",
            "includePath": ["$rootdir/Include","$rootdir/Include/Core"$includelist],
            "cStandard": "c99",
            "cppStandard": "c++14"
        }
    ],
    "version": 4
}
EOF
    close(FH);

    $optfile = "$rootdir/.vscode/settings.json";
    open(FH, '>', $optfile) or die $!;
    print FH <<EOF;
{
    "files.associations": {
        "*.h": "c",
        "*.c": "c",
        "*.[hc].sdtmpl": "c",
        "*.h.tmpl": "c",
        "*.hpp": "cuda",
        "*.cu": "cuda",
        "*.cu.sdtmpl": "cuda",
    }
}
EOF
    close(FH);

}

# this is needed
1;

