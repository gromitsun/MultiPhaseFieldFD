CC = clang++
CFLAGS = -Wall -O3 -std=c++11 -I../../cl_common/ -I../../sim_common/ -DDEBUG
LFLAGS = -framework OpenCL -L../../cl_common/ -lclcommon -L../../sim_common/ -lsimcommon
EXE = main.out
SUB_DIR = one_phi_local/


all: $(EXE)
	export DYLD_LIBRARY_PATH=$$DYLD_LIBRARY_PATH:../../cl_common/:../../sim_common/
$(EXE): $(OBJECTS) $(SUB_OBJECTS)
	$(CC) -o $(EXE) $(OBJECTS) $(SUB_OBJECTS) $(LFLAGS)
$(OBJECTS): %.o :  %.cpp
	$(CC) $(CFLAGS) -c $< -I$(SUB_DIR) -I. -o $@
$(SUB_OBJECTS): $(SUB_DIR)/%.o : $(SUB_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -I$(SUB_DIR) -I. -o $@
clean:
	rm -fv $(EXE)
	rm -fv *.o
	rm -fv *.gch