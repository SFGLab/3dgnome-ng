CC      = g++
CFLAGS  = -std=c++0x -Wno-write-strings -fPIC -O2

MC_DIR      = 3dnome/MC
MC_SRCS_CPP = $(wildcard $(MC_DIR)/*.cpp) $(wildcard $(MC_DIR)/lib/*.cpp)
MC_SRCS_C   = $(wildcard $(MC_DIR)/lib/*.c)
MC_SRCS     = $(MC_SRCS_CPP) $(MC_SRCS_C)

# macOS uses -install_name / @executable_path; Linux uses -soname / $ORIGIN
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    SONAME_FLAG = -Wl,-install_name,@rpath/lib3dnome.so
    RPATH_FLAG  = -Wl,-rpath,@executable_path
else
    SONAME_FLAG = -Wl,-soname,lib3dnome.so
    RPATH_FLAG  = -Wl,-rpath,$$ORIGIN
endif

.PHONY: all 3dnome scorer clean

all: 3dnome scorer

# Build 3dnome here (not via 3dnome/makefile) so linker flags are OS-aware.
3dnome: 3dnome/lib3dnome.so 3dnome/3dnome

3dnome/lib3dnome.so: $(MC_SRCS)
	$(CC) $(CFLAGS) -shared $(SONAME_FLAG) $^ -o $@

3dnome/3dnome: $(MC_DIR)/tools/main.cpp 3dnome/lib3dnome.so
	$(CC) $(CFLAGS) $< -o $@ -l3dnome -I$(MC_DIR) -L3dnome/ $(RPATH_FLAG)

scorer: harness/scorer

harness/scorer: harness/scorer.cpp $(MC_SRCS)
	$(CC) $(CFLAGS) -I$(MC_DIR) -o $@ $^ -lm

clean:
	rm -f 3dnome/lib3dnome.so 3dnome/3dnome harness/scorer
