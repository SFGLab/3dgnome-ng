CC      = g++
CFLAGS  = -std=c++0x -Wno-write-strings -fPIC -O2

MC_DIR      = 3dnome/MC
MC_SRCS_CPP = $(wildcard $(MC_DIR)/*.cpp) $(wildcard $(MC_DIR)/lib/*.cpp)
MC_SRCS_C   = $(wildcard $(MC_DIR)/lib/*.c)
MC_SRCS     = $(MC_SRCS_CPP) $(MC_SRCS_C)

.PHONY: all 3dnome scorer clean

all: 3dnome scorer

3dnome:
	$(MAKE) -C 3dnome

scorer: harness/scorer

harness/scorer: harness/scorer.cpp $(MC_SRCS)
	$(CC) $(CFLAGS) -I$(MC_DIR) -o $@ $^ -lm

clean:
	$(MAKE) -C 3dnome clean
	rm -f harness/scorer
