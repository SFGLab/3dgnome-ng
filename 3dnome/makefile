CC      = g++
CFLAGS  = -fPIC -std=c++0x -Wno-write-strings

SRC_CPP = $(wildcard MC/*.cpp) $(wildcard MC/lib/*.cpp)
SRC_C   = $(wildcard MC/lib/*.c)
SRC     = $(SRC_CPP) $(SRC_C)

.PHONY: all clean

all: lib3dnome.so 3dnome

lib3dnome.so: $(SRC)
	$(CC) $(CFLAGS) -shared -Wl,-install_name,@rpath/lib3dnome.so $^ -o $@

3dnome: MC/tools/main.cpp lib3dnome.so
	$(CC) $(CFLAGS) $< -o $@ -l3dnome -I"./MC" -L"./" -Wl,-rpath,@executable_path

clean:
	rm -f lib3dnome.so 3dnome
