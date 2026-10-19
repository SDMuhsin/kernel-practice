# Lesson 3 - CUDA. Plain .cu files, nvcc, no notebooks.
#
#   make            list the exercises
#   make 01         build and run exercise 01
#   make 01s        build and run the solution
#   make all        every exercise
#   make solutions  every solution (the regression test)
#   make clean
#
# env.sh sets NVCC, SM, CUDA_INC and the link flags for you. Run
# `bash scripts/doctor.sh` if a build fails.

NVCC     ?= $(if $(KATA_NVCC),$(KATA_NVCC),nvcc)
SM       ?= $(KATA_SM)
ARCH     := $(if $(SM),-arch=sm_$(SM),)
# nvcc's own headers first, then a system CUDA only to fill in libcu++
NVROOT   := $(patsubst %/bin/nvcc,%,$(NVCC))
INC      := -Iinclude -I$(NVROOT)/include $(if $(KATA_CUDA_INC),-I$(KATA_CUDA_INC),)
# a pip CUDA wheel ships no libcudart.so symlink, so link the static runtime
LIBS     := $(if $(wildcard $(NVROOT)/lib/libcudart_static.a),\
              -L$(NVROOT)/lib -lcudart_static -lrt -lpthread -ldl,-lcudart)
NVFLAGS  ?= -std=c++17 -O2 -lineinfo $(ARCH) $(INC)
BUILD    ?= $(if $(KATA_BUILD),$(KATA_BUILD),/tmp/kata-build)/cuda

DIRS := $(sort $(wildcard [0-9][0-9]_*))
NUMS := $(foreach d,$(DIRS),$(firstword $(subst _, ,$d)))

.PHONY: help all solutions clean fixture $(NUMS) $(addsuffix s,$(NUMS))

help:
	@echo "Lesson 3 - CUDA"
	@echo
	@$(foreach d,$(DIRS), printf '  make %-4s %s\n' \
	   "$(firstword $(subst _, ,$d))" "$(d)";)
	@echo
	@echo "  make all        run every exercise"
	@echo "  make solutions  run every solution"
	@echo "  make fixture    generate the capstone data (needed once, for 13+)"
	@echo
	@echo "Open the .cu file, read the header, fill in the TODOs, then run it."

define RULE
$(firstword $(subst _, ,$(1))): $(BUILD)/$(1)-exercise
	@echo "--- $(1)/exercise.cu ---"
	@$$<
$(firstword $(subst _, ,$(1)))s: $(BUILD)/$(1)-solution
	@echo "--- $(1)/solution.cu ---"
	@$$<
$(BUILD)/$(1)-exercise: $$(wildcard $(1)/*.cu) include/check.cuh include/fixture.cuh
	@mkdir -p $(BUILD)
	@$(NVCC) $(NVFLAGS) -I$(1) -o $$@ $(1)/exercise.cu $(LIBS)
$(BUILD)/$(1)-solution: $$(wildcard $(1)/*.cu) include/check.cuh include/fixture.cuh
	@mkdir -p $(BUILD)
	@$(NVCC) $(NVFLAGS) -I$(1) -o $$@ $(1)/solution.cu $(LIBS)
endef
$(foreach d,$(DIRS),$(eval $(call RULE,$(d))))

fixture:
	@python 13_gemv/gen_fixture.py

all:
	@fail=0; for n in $(NUMS); do $(MAKE) -s $$n || fail=1; echo; done; exit $$fail

solutions:
	@fail=0; for n in $(NUMS); do $(MAKE) -s $${n}s || fail=1; echo; done; \
	 if [ $$fail -eq 0 ]; then echo "all solutions pass"; else echo "SOME SOLUTIONS FAIL"; fi; \
	 exit $$fail

clean:
	@rm -rf $(BUILD)
	@echo "removed $(BUILD)"
