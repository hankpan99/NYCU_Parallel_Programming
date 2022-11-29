all: matmul

matmul: main.o matmul.o
	mpicxx -O3 -o matmul main.o matmul.o

matmul.o: main.o
	mpicxx -O3 -c matmul.cc

main.o: main.cc
	mpicxx -O3 -c main.cc

clean:
	rm -f matmul *.o

# all:
# 	mpicxx -o matmul main.cc matmul.cc

# clean:
# 	rm -f matmul