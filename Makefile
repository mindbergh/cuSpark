OBJDIR=objs
SRCDIR=src
TESTDIR=test
LOGIDR=logs.*
LIBDIR=libs

.PHONY: all clean test
all :
test :

SHELL := /bin/bash

LIBS=libglog libgflags
PKGCONFIG=PKG_CONFIG_PATH=$(LIBDIR)/pkgconfig:$$PKG_CONFIG_PATH pkg-config

LIBS_NOTFOUND:=$(foreach lib, $(LIBS), $(shell $(PKGCONFIG) --exists $(lib); if [ $$? -ne 0 ]; then echo $(lib); fi))
ifneq ($(strip $(LOBS_NOTFOUND)),)
	$(error "Libraries not found: '$(LIBS_NOTFOUND'")
endif

CXX=g++
CXXFLAGS+=-Wall -Wexgtra -O2
CPPFLAGS+=-I$(CURDIR)/$(SRCDIR) $(foreach lib, $(LIBS), $(shell $PKGCONFIG) --cflags $(lib)))
LDFLAGS+= $(foreach lib,$(LIBS), $(shell $(PKGCONFIG) --libs $(lib))) -Xlinker -rpath -Xlinker $(LIBDIR)

$(LOGDIR):
	mkdir -p $@

local_server: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJDIR)

test:



