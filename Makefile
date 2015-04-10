LIBS=libs
SRC=src
OBJ=obj


MODULES		:= pipeline common test
SRC_DIR   := $(addprefix $(SRC)/,$(MODULES))
OBJ_DIR 	:= $(addprefix $(OBJ)/,$(MODULES))

SRCS			:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cc))
OBJS			:= $(patsubst $(SRC)/%.cc,$(OBJ)/%.o,$(SRCS))
INCLUDES	+= -I./ -I$(LIBS)/include -I$(SRC)

CXX = g++
CXXFLAGS = -g -std=c++0x -L$(LIBS) -lgtest -lglog -lgflags -lpthread

vpath %.cc $(SRC_DIR)

define make-goal
$1/%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $$< -o $$@
endef

.PHONY: all test checkdirs clean

all: checkdirs testall


testall: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o testall $(SRC)/testall/testall.cc $(OBJS)

print-%  : ; @echo $* = $($*)

checkdirs: $(OBJ_DIR)

$(OBJ_DIR):
	@mkdir -p $@

clean:
	rm -r obj

$(foreach odir,$(OBJ_DIR),$(eval $(call make-goal,$(odir))))




