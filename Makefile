OBJDIR=objs
SRCDIR=src
TESTDIR=test
LOGIDR=logs.*
LIBDIR=libs
GTEST_DIR=$(LIBDIR)/gtest-1.7.0
TESTS=pipeline_test

.PHONY: all clean test
all : 

# Define our wonderful make functions.
include functions.mk
include 

# Include all sub directories in the following way
# $(eval $(call define_program,worker,     \
#         $(SRCDIR)/worker/main.cpp        \
#         $(SRCDIR)/worker/work_engine.cpp \
# ))

SHELL := /bin/bash

LIBS=libglog libgflags
PKGCONFIG=PKG_CONFIG_PATH=$(LIBDIR)/pkgconfig:$$PKG_CONFIG_PATH pkg-config

LIBS_NOTFOUND:=$(foreach lib, $(LIBS), $(shell $(PKGCONFIG) --exists $(lib); if [ $$? -ne 0 ]; then echo $(lib); fi))
ifneq ($(strip $(LOBS_NOTFOUND)),)
	$(error "Libraries not found: '$(LIBS_NOTFOUND'")
endif

CXX=g++
CXXFLAGS += -g -Wall -Wextra -O2
CPPFLAGS += -isystem $(GTEST_DIR)/include -I$(CURDIR)/$(SRCDIR)/pipeline $(foreach lib,$(LIBS), $(shell $(PKGCONFIG) --cflags $(lib)))

$(LOGDIR):
	mkdir -p $@

local_server: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -c -o $@

GTEST_HEADERS=$(GTEST_DIR)/include/gtest/*.h# $(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
$(GTEST_DIR)/obj/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
         	$(GTEST_DIR)/src/gtest-all.cc -o $(GTEST_DIR)/obj/gtest-all.o

$(GTEST_DIR)/obj/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            	$(GTEST_DIR)/src/gtest_main.cc -o $(GTEST_DIR)/obj/gtest_main.o

$(GTEST_DIR)/obj/gtest.a : $(GTEST_DIR)/obj/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_DIR)/obj/gtest_main.a : $(GTEST_DIR)/obj/gtest-all.o $(GTEST_DIR)/obj/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_DIR)/obj/pipeline_test.o: $(TESTDIR)/pipeline_test.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(TESTDIR)/pipeline_test.cc
	
pipeline_test: $(GTEST_DIR)/obj/gtest_main.a $(GTEST_DIR)/obj/pipeline_test.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@

clean:
	rm -rf $(OBJDIR)

test: $(TESTS)
	echo $(TESTS)
