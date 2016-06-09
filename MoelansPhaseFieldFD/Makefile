CC = clang++
CFLAGS = -Wall -O3 -std=c++11 -I../cl_common/ -I../sim_common/ -DDEBUG
LFLAGS = -framework OpenCL -L../cl_common/ -lclcommon -L../sim_common/ -lsimcommon
EXE = main.out
SRC_FILES = $(wildcard *.cpp)
OBJ_FILES = $(SRC_FILES:.cpp=.o)


all: $(EXE)
	export DYLD_LIBRARY_PATH=$$DYLD_LIBRARY_PATH:../cl_common/:../sim_common/
$(EXE): $(OBJ_FILES)
	$(CC) -o $(EXE) $(OBJ_FILES) $(LFLAGS)
$(OBJ_FILES): $(SRC_FILES)
	$(CC) -c $(SRC_FILES) $(CFLAGS)
clean:
	rm -fv $(EXE)
	rm -fv *.o
	rm -fv *.gch
