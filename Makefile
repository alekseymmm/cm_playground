TOP_DIR ?= $(shell pwd)

KDIR ?= /lib/modules/$(shell uname -r)/build

CC := gcc

TOPTARGETS := all clean
SUBDIRS := $(wildcard */.)

export TOP_DIR

$(TOPTARGETS): $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: $(TOPTARGETS) $(SUBDIRS)
