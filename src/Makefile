top: clean program

program: fitness classifiers
	g++ program.cpp -std=c++11 -g -o program -lpthread fitness.o classifiers.o

fitness: fitness.cpp fitness.hpp
	g++ fitness.cpp -std=c++11 -g -c 

classifiers: classifiers.cpp classifiers.hpp
	g++ classifiers.cpp -std=c++11 -g -c

run_program: clean program
	@./program ../data/small/blogger/blogger.data


tests: fitness 
	g++ instance_selection_unittest.cpp -std=c++11 -I./gtest-1.7.0/include \
			gtest-1.7.0/src/gtest-all.o -o instance_selection_unittest -lpthread fitness.o

run_tests: tests
	@./instance_selection_unittest

clean:
	@rm *.o || :
	@rm program || :
	@rm -r program.dSYM/ || :
