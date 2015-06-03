top: clean program

program: testing
	g++ program.cpp -std=c++11 -O2 -o program -pthread testing.o fitness.o classifiers.o

fitness: fitness.cpp fitness.hpp
	g++ fitness.cpp -std=c++11 -pthread -O2 -c

classifiers: classifiers.cpp classifiers.hpp
	g++ classifiers.cpp -std=c++11 -pthread -O2 -c

testing: testing.cpp testing.hpp fitness classifiers
	g++ testing.cpp fitness.o classifiers.o -std=c++11 -pthread -O2 -c 

run_program: clean program
	@./program ../data/small/glass/glass.data euclidean


tests: fitness
	g++ instance_selection_unittest.cpp -std=c++11 -I./gtest-1.7.0/include \
			gtest-1.7.0/src/gtest-all.o -o instance_selection_unittest -lpthread fitness.o

run_tests: tests
	@./instance_selection_unittest

clean:
	@rm *.o || :
	@rm *.gch || :
	@rm program || :
	@rm -r program.dSYM/ || :
