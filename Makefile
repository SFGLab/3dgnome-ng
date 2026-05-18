CC      = g++
CFLAGS  = -std=c++17 -Wno-write-strings -fPIC -O2

MC_DIR      = 3dnome/MC
MC_SRCS_CPP = $(wildcard $(MC_DIR)/*.cpp) $(wildcard $(MC_DIR)/lib/*.cpp)
MC_SRCS_C   = $(wildcard $(MC_DIR)/lib/*.c)
MC_SRCS     = $(MC_SRCS_CPP) $(MC_SRCS_C)

.PHONY: all 3dnome scorer clean

all: 3dnome scorer

# Link 3dnome as a single static binary (all MC sources + main) to avoid
# shared-library / rpath differences between macOS and Linux.
3dnome: 3dnome/3dnome

3dnome/3dnome: $(MC_DIR)/tools/main.cpp $(MC_SRCS)
	$(CC) $(CFLAGS) -I$(MC_DIR) -o $@ $^

scorer: harness/scorer

harness/scorer: harness/scorer.cpp $(MC_SRCS)
	$(CC) $(CFLAGS) -I$(MC_DIR) -o $@ $^ -lm

clean:
	rm -f 3dnome/3dnome harness/scorer
