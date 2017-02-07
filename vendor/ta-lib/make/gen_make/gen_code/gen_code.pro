TEMPLATE = app
CONFIG   -= qt

# Force this application to be a console application
CONFIG   -= windows
CONFIG   += console

# Identify the temp dir
cmd:OBJECTS_DIR = ../../../../../temp/cmd/gen_code
cmr:OBJECTS_DIR = ../../../../../temp/cmr/gen_code
cmp:OBJECTS_DIR = ../../../../../temp/cmp/gen_code
csd:OBJECTS_DIR = ../../../../../temp/csd/gen_code
csr:OBJECTS_DIR = ../../../../../temp/csr/gen_code
csp:OBJECTS_DIR = ../../../../../temp/csp/gen_code
cdr:OBJECTS_DIR = ../../../../../temp/cdr/gen_code
cdd:OBJECTS_DIR = ../../../../../temp/cdd/gen_code

# Output info
TARGET      = gen_code
DESTDIR     = ../../../../../bin

# Files to process

SOURCES = ../../../../../src/tools/gen_code/gen_code.c \
          ../../../../../src/ta_abstract/ta_abstract.c \
          ../../../../../src/ta_abstract/ta_def_ui.c \
          ../../../../../src/ta_abstract/tables/table_a.c \
          ../../../../../src/ta_abstract/tables/table_b.c \
          ../../../../../src/ta_abstract/tables/table_c.c \
          ../../../../../src/ta_abstract/tables/table_d.c \
          ../../../../../src/ta_abstract/tables/table_e.c \
          ../../../../../src/ta_abstract/tables/table_f.c \
          ../../../../../src/ta_abstract/tables/table_g.c \
          ../../../../../src/ta_abstract/tables/table_h.c \
          ../../../../../src/ta_abstract/tables/table_i.c \
          ../../../../../src/ta_abstract/tables/table_j.c \
          ../../../../../src/ta_abstract/tables/table_k.c \
          ../../../../../src/ta_abstract/tables/table_l.c \
          ../../../../../src/ta_abstract/tables/table_m.c \
          ../../../../../src/ta_abstract/tables/table_n.c \
          ../../../../../src/ta_abstract/tables/table_o.c \
          ../../../../../src/ta_abstract/tables/table_p.c \
          ../../../../../src/ta_abstract/tables/table_q.c \
          ../../../../../src/ta_abstract/tables/table_r.c \
          ../../../../../src/ta_abstract/tables/table_s.c \
          ../../../../../src/ta_abstract/tables/table_t.c \
          ../../../../../src/ta_abstract/tables/table_u.c \
          ../../../../../src/ta_abstract/tables/table_v.c \
          ../../../../../src/ta_abstract/tables/table_w.c \
          ../../../../../src/ta_abstract/tables/table_x.c \
          ../../../../../src/ta_abstract/tables/table_y.c \
          ../../../../../src/ta_abstract/tables/table_z.c


# Additional libraries
win32:TA_COMMON_CMD = ta_common_cmd.lib
win32:TA_COMMON_CMR = ta_common_cmr.lib
win32:TA_COMMON_CSD = ta_common_csd.lib
win32:TA_COMMON_CSR = ta_common_csr.lib
win32:TA_COMMON_CDR = ta_common_cdr.lib
win32:TA_COMMON_CDD = ta_common_cdd.lib

unix:TA_COMMON_CMD  = libta_common_cmd.a
unix:TA_COMMON_CMR  = libta_common_cmr.a
unix:TA_COMMON_CMP  = libta_common_cmp.a
unix:TA_COMMON_CSD  = libta_common_csd.a
unix:TA_COMMON_CSR  = libta_common_csr.a
unix:TA_COMMON_CSP  = libta_common_csp.a
unix:TA_COMMON_CDR  = libta_common_cdr.a
unix:TA_COMMON_CDD  = libta_common_cdd.a

cmd:LIBS += ../../../../../lib/$$TA_COMMON_CMD
cmr:LIBS += ../../../../../lib/$$TA_COMMON_CMR
cmp:LIBS += ../../../../../lib/$$TA_COMMON_CMP
csd:LIBS += ../../../../../lib/$$TA_COMMON_CSD
csr:LIBS += ../../../../../lib/$$TA_COMMON_CSR
csp:LIBS += ../../../../../lib/$$TA_COMMON_CSP
cdr:LIBS += ../../../../../lib/$$TA_COMMON_CDR
cdd:LIBS += ../../../../../lib/$$TA_COMMON_CDD

unix:LIBS += -ldl

# Compiler Options
INCLUDEPATH = ../../../../../include \
              ../../../../../src/ta_common \
              ../../../../../src/ta_abstract \
              ../../../../../src/ta_abstract/tables \
              ../../../../../src/ta_abstract/frames

DEFINES *= TA_GEN_CODE

# debug/release dependent options.
debug:DEFINES  *= TA_DEBUG
debug:DEFINES  *= _DEBUG
DEFINES        += TA_SINGLE_THREAD
thread:DEFINES -= TA_SINGLE_THREAD


# Platform dependent options.
win32:DEFINES              *= WIN32
win32-msvc:DEFINES         *= _MBCS _LIB
cygwin-g++:LIBS            -= -ldl
freebsd-g++:LIBS      -= -ldl
freebsd-g++:INCLUDEPATH += /usr/local/include

cmd:TEMP_CLEAN_ALL = ../../../../../temp/cmd/*.pch
cmr:TEMP_CLEAN_ALL = ../../../../../temp/cmr/*.pch
cmp:TEMP_CLEAN_ALL = ../../../../../temp/cmp/*.pch
csd:TEMP_CLEAN_ALL = ../../../../../temp/csd/*.pch
csr:TEMP_CLEAN_ALL = ../../../../../temp/csr/*.pch
csp:TEMP_CLEAN_ALL = ../../../../../temp/csp/*.pch
cdr:TEMP_CLEAN_ALL = ../../../../../temp/cdr/*.pch
cdd:TEMP_CLEAN_ALL = ../../../../../temp/cdd/*.pch

cmd:TEMP_CLEAN_ALL2 = ../../../../../temp/cmd/*.idb
cmr:TEMP_CLEAN_ALL2 = ../../../../../temp/cmr/*.idb
cmp:TEMP_CLEAN_ALL2 = ../../../../../temp/cmp/*.idb
csd:TEMP_CLEAN_ALL2 = ../../../../../temp/csd/*.idb
csr:TEMP_CLEAN_ALL2 = ../../../../../temp/csr/*.idb
csp:TEMP_CLEAN_ALL2 = ../../../../../temp/csp/*.idb
cdr:TEMP_CLEAN_ALL2 = ../../../../../temp/cdr/*.idb
cdd:TEMP_CLEAN_ALL2 = ../../../../../temp/cdd/*.idb

cmd:TEMP_CLEAN_ALL3 = ../../../../../temp/cmd/$$TARGET/*.pch
cmr:TEMP_CLEAN_ALL3 = ../../../../../temp/cmr/$$TARGET/*.pch
cmp:TEMP_CLEAN_ALL3 = ../../../../../temp/cmp/$$TARGET/*.pch
csd:TEMP_CLEAN_ALL3 = ../../../../../temp/csd/$$TARGET/*.pch
csr:TEMP_CLEAN_ALL3 = ../../../../../temp/csr/$$TARGET/*.pch
csp:TEMP_CLEAN_ALL3 = ../../../../../temp/csp/$$TARGET/*.pch
cdr:TEMP_CLEAN_ALL3 = ../../../../../temp/cdr/$$TARGET/*.pch
cdd:TEMP_CLEAN_ALL3 = ../../../../../temp/cdd/$$TARGET/*.pch

cmd:TEMP_CLEAN_ALL4 = ../../../../../temp/cmd/$$TARGET/*.idb
cmr:TEMP_CLEAN_ALL4 = ../../../../../temp/cmr/$$TARGET/*.idb
cmp:TEMP_CLEAN_ALL4 = ../../../../../temp/cmp/$$TARGET/*.idb
csd:TEMP_CLEAN_ALL4 = ../../../../../temp/csd/$$TARGET/*.idb
csr:TEMP_CLEAN_ALL4 = ../../../../../temp/csr/$$TARGET/*.idb
csp:TEMP_CLEAN_ALL4 = ../../../../../temp/csp/$$TARGET/*.idb
cdr:TEMP_CLEAN_ALL4 = ../../../../../temp/cdr/$$TARGET/*.idb
cdd:TEMP_CLEAN_ALL4 = ../../../../../temp/cdd/$$TARGET/*.idb

cmd:TEMP_CLEAN_ALL5 = ../../../../../temp/cmd/$$TARGET/*.obj
cmr:TEMP_CLEAN_ALL5 = ../../../../../temp/cmr/$$TARGET/*.obj
cmp:TEMP_CLEAN_ALL5 = ../../../../../temp/cmp/$$TARGET/*.obj
csd:TEMP_CLEAN_ALL5 = ../../../../../temp/csd/$$TARGET/*.obj
csr:TEMP_CLEAN_ALL5 = ../../../../../temp/csr/$$TARGET/*.obj
csp:TEMP_CLEAN_ALL5 = ../../../../../temp/csp/$$TARGET/*.obj
cdr:TEMP_CLEAN_ALL5 = ../../../../../temp/cdr/$$TARGET/*.obj
cdd:TEMP_CLEAN_ALL5 = ../../../../../temp/cdd/$$TARGET/*.obj

win32:CLEAN_FILES = ../../../../../bin/*.map ../../../../../bin/*._xe ../../../../../bin/*.tds ../../../../../bin/*.pdb ../../../../../bin/*.pbo ../../../../../bin/*.pbi ../../../../../bin/*.pbt $$TEMP_CLEAN_ALL $$TEMP_CLEAN_ALL2 $$TEMP_CLEAN_ALL3 $$TEMP_CLEAN_ALL4 $$TEMP_CLEAN_ALL5
