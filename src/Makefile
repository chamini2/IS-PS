top: clean principal

principal: fitness classifiers
	g++ principal.cpp -std=c++11 -g -o principal -lpthread fitness.o classifiers.o

abalone: fitness
	g++ abalone.cpp -std=c++11 -g -o abalone -lpthread fitness.o -O2

fitness: fitness.cpp fitness.hpp
	g++ fitness.cpp -std=c++11 -g -c 

classifiers: classifiers.cpp classifiers.hpp
	g++ classifiers.cpp -std=c++11 -g -c

run: top
	@./abalone ../data/abalone/abalone.data

run_principal: clean principal
	@./principal ../data/small/blogger/blogger.data


tests: fitness 
	g++ instance_selection_unittest.cpp -std=c++11 -I./gtest-1.7.0/include \
			gtest-1.7.0/src/gtest-all.o -o instance_selection_unittest -lpthread fitness.o

run_tests: tests
	@./instance_selection_unittest

clean:
	@rm *.o || rm principal || rm abalone || :
