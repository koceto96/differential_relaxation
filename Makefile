#!/bin/make
SHELL= /bin/bash
C= gcc
CC= cc
CFLAGS= -O3 -g3 -Wall -Werror -Wextra -pedantic -std=c11 -lpthread
CP= cp
RM= rm
CHMOD= chmod

INSTALL= install

TARGETS= parallel-c1

all: clobber ${TARGETS}

parallel-c1: parallel_c1.c
	${C} ${CFLAGS} parallel_c1.c -o $@

test: all
	@echo Starting test suite
	@echo
	@for test_file in $(shell ls test/); do\
		for threads in $(shell seq 2 8); do \
			for i in $(shell seq 0 10); do \
			    echo ./parallel-c1 -f test/$$test_file -d $$(cat test/$$test_file | wc -l) $$threads $$(awk -v n=$$i 'BEGIN {printf("0."); for(i = 0; i < n; i++) printf("0"); printf("1")}') \
				./parallel-c1 -f test/$$test_file -d $$(cat test/$$test_file | wc -l) $$threads $$(awk -v n=$$i 'BEGIN {printf("0."); for(i = 0; i < n; i++) printf("0"); printf("1")}'); done; done; done
	@echo All tests finished successfully!

configure:
	@echo nothing to configure

clean quick_clean quick_distclean distclean:

clobber quick_clobber: clean
	${RM} -f ${TARGETS}

install: all
