# Lesson 2 - C++. Plain files, a real compiler, no notebooks.
#
#   make            list the exercises
#   make 01         build and run exercise 01
#   make 01s        build and run the solution to 01
#   make all        run every exercise (expect failures until you fill them in)
#   make solutions  run every solution (this is the regression test)
#   make clean
#
# Set CXX if you want clang++ instead of g++.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O1 -g -Wall -Wextra -Iinclude
BUILD    ?= $(if $(KATA_BUILD),$(KATA_BUILD),/tmp/kata-build)/cpp

DIRS  := $(sort $(wildcard [0-9][0-9]_*))
NUMS  := $(foreach d,$(DIRS),$(firstword $(subst _, ,$d)))

.PHONY: help all solutions clean $(NUMS) $(addsuffix s,$(NUMS))

help:
	@echo "Lesson 2 - C++"
	@echo
	@$(foreach d,$(DIRS), printf '  make %-4s %s\n' \
	   "$(firstword $(subst _, ,$d))" "$(d)";)
	@echo
	@echo "  make all        run every exercise"
	@echo "  make solutions  run every solution"
	@echo
	@echo "Open the .cpp file, read the header, fill in the TODOs, then run it."

# `make 01` -> build and run 01_*/exercise.cpp
define RULE
$(firstword $(subst _, ,$(1))): $(BUILD)/$(1)-exercise
	@echo "--- $(1)/exercise.cpp ---"
	@$$<
$(firstword $(subst _, ,$(1)))s: $(BUILD)/$(1)-solution
	@echo "--- $(1)/solution.cpp ---"
	@$$<
$(BUILD)/$(1)-exercise: $$(wildcard $(1)/*.cpp) $$(wildcard $(1)/*.h) include/check.h
	@mkdir -p $(BUILD)
	@$(CXX) $(CXXFLAGS) -I$(1) -o $$@ $(1)/exercise.cpp \
	   $$(filter-out $(1)/exercise.cpp $(1)/solution.cpp,$$(wildcard $(1)/*.cpp))
$(BUILD)/$(1)-solution: $$(wildcard $(1)/*.cpp) $$(wildcard $(1)/*.h) include/check.h
	@mkdir -p $(BUILD)
	@$(CXX) $(CXXFLAGS) -I$(1) -o $$@ $(1)/solution.cpp \
	   $$(filter-out $(1)/exercise.cpp $(1)/solution.cpp,$$(wildcard $(1)/*.cpp))
endef
$(foreach d,$(DIRS),$(eval $(call RULE,$(d))))

all:
	@fail=0; for n in $(NUMS); do $(MAKE) -s $$n || fail=1; echo; done; exit $$fail

solutions:
	@fail=0; for n in $(NUMS); do $(MAKE) -s $${n}s || fail=1; echo; done; \
	 if [ $$fail -eq 0 ]; then echo "all solutions pass"; else echo "SOME SOLUTIONS FAIL"; fi; \
	 exit $$fail

clean:
	@rm -rf $(BUILD)
	@echo "removed $(BUILD)"
