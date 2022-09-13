update:
	rsync -avz /uufs/chpc.utah.edu/common/home/u6039184/epiworld/epiworld.hpp .

main.o: 00-sirconn.cpp epiworld.hpp bmark.hpp
	g++ -std=c++17 -mtune=native -O3 -Wall -pedantic 00-sirconn.cpp -o main.o
