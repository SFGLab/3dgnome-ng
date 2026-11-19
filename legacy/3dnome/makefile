CC      = g++
CFLAGS  = -fPIC -std=c++0x -Wno-write-strings

SRC_CPP = $(wildcard MC/*.cpp) $(wildcard MC/lib/*.cpp)
SRC_C   = $(wildcard MC/lib/*.c)
SRC     = $(SRC_CPP) $(SRC_C)

# The shared-library install name and the runtime search path are spelled differently by the two
# linkers. Darwin ld uses install_name with the @rpath and @executable_path tokens. GNU ld uses
# soname with the $ORIGIN token. Selecting per platform lets the same makefile build on both.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SONAME  = -Wl,-install_name,@rpath/lib3dnome.so
  RPATH   = -Wl,-rpath,@executable_path
else
  SONAME  = -Wl,-soname,lib3dnome.so
  RPATH   = -Wl,-rpath,'$$ORIGIN'
endif

.PHONY: all clean

all: lib3dnome.so 3dnome

lib3dnome.so: $(SRC)
	$(CC) $(CFLAGS) -shared $(SONAME) $^ -o $@

3dnome: MC/tools/main.cpp lib3dnome.so
	$(CC) $(CFLAGS) $< -o $@ -l3dnome -I"./MC" -L"./" $(RPATH)

clean:
	rm -f lib3dnome.so 3dnome
