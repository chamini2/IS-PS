top: clean balloons

balloons: fitness
	g++ balloons.cpp -std=c++11 -g -o balloons -I./gflags/2.1.2/include/ \
		./gflags-2.1.2/lib/libgflags.a ./gflags-2.1.2/lib/libgflags_nothreads.a -lpthread fitness.o

fitness: fitness.cpp fitness.hpp
	g++ fitness.cpp -std=c++11 -g -c 

run: balloons
	@./balloons --data_file=../data/balloons/adult-stretch.data --points_to_toggle=3 --local_iterations=100

tests:
	g++ instance_selection_unittest.cpp -std=c++11 -I./gtest-1.7.0/include \
			gtest-1.7.0/src/gtest-all.o -o instance_selection_unittest -lpthread

run_tests: tests
	@./instance_selection_unittest

clean:
	@rm balloons || :
