#!/bin/bash
. .pack/packrc

##########################################################
#C++
##########################################################
#Compile
TYPE=TESTING make tests/test_Module.tout

#Run module testing
TYPE=TESTING make tests/test_Module.tout_run

#Run integration testing
bash tests/test_cpp_integration.sh

#Coverage
gcovr -r .


##########################################################
#PYTHON
##########################################################
#Run unitary tests
$NOSETESTS --with-coverage --cover-package=. 

#Run integration tests
#bash tests/test_python_integration.sh

