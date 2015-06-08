top: clean program

program: testing
	g++ program.cpp instance_selection.o fitness.o classifiers.o testing.o -std=c++11 -g -o program -pthread ${OPTIMIZATION_FLAGS} 

fitness: fitness.cpp fitness.hpp
	g++ fitness.cpp -std=c++11 -pthread -g -c ${OPTIMIZATION_FLAGS}

classifiers: classifiers.cpp classifiers.hpp
	g++ classifiers.cpp -std=c++11 -pthread -g -c ${OPTIMIZATION_FLAGS}  

testing: testing.cpp testing.hpp instance_selection
	g++ testing.cpp -std=c++11 -pthread -g -c ${OPTIMIZATION_FLAGS} 

instance_selection: instance_selection.cpp instance_selection.hpp fitness classifiers 
	g++ instance_selection.cpp -std=c++11 -pthread -g -c ${OPTIMIZATION_FLAGS} 

run_ls: clean program
	@./program ls_tests.txt

run_ils_grasp: clean program
	@./program ils_grasp_tests.txt

tests: fitness instance_selection classifiers
	g++ instance_selection_unittest.cpp -std=c++11 -I./gtest-1.7.0/include \
			gtest-1.7.0/src/gtest-all.o -o instance_selection_unittest -pthread fitness.o instance_selection.o classifiers.o ${OPTIMIZATION_FLAGS}

run_tests: tests
	@./instance_selection_unittest

clean:
	@rm *.o || :
	@rm *.gch || :
	@rm program || :
	@rm -r program.dSYM/ || :
