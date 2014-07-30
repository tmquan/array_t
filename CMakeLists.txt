# ----------------------------------------------------------------------------
# Top level Wavelet Neural Network
# Author		:	Tran Minh Quan
# Email			: 	quantm@unist.ac.kr,	quantm@kaist.ac.kr
# Address		: 	High-performance Visual Computing Lab
# Website		:	http://hvcl.unist.ac.kr/
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# ----------------------------------------------------------------------------
project(DeepNeuralNetwork CXX C)

# ----------------------------------------------------------------------------
# All binary files will be stored here
# ----------------------------------------------------------------------------
# set(RESOURCE_INPUT_DIRECTORY 		"${CMAKE_SOURCE_DIR}/data")
# set(RUNTIME_OUTPUT_DIRECTORY 		"${CMAKE_SOURCE_DIR}/bin")
# set(LIBRARY_OUTPUT_DIRECTORY 		"${CMAKE_SOURCE_DIR}/lib")
# set(INCLUDE_HEADER_DIRECTORY 		"${CMAKE_SOURCE_DIR}/include")

set(RESOURCE_INPUT_DIRECTORY 		"${CMAKE_BINARY_DIR}/data")
set(RUNTIME_OUTPUT_DIRECTORY 		"${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_DIRECTORY 		"${CMAKE_BINARY_DIR}/lib")
set(INCLUDE_HEADER_DIRECTORY 		"${CMAKE_BINARY_DIR}/include")

# ----------------------------------------------------------------------------
# Enable nightly test
# ----------------------------------------------------------------------------
enable_testing()

# ----------------------------------------------------------------------------
# Process subdirectories 
# ----------------------------------------------------------------------------
# dnn.hpp and legacy headers
# add_subdirectory(include)

# libraries directory 
# add_subdirectory(lib)

# Generate targets for documentation 
# add_subdirectory(doc)

# Various data that is used by  libraries and/or demo applications.
# add_subdirectory(data)

# modules
add_subdirectory(modules)

# Examples
add_subdirectory(samples)

# ----------------------------------------------------------------------------
# Clean the output and test logs
# ----------------------------------------------------------------------------
add_custom_target(clear 
	COMMAND rm			
	`find ./ -iname '*Make*' -not -name CMakeLists.txt` 
	`find ./ -iname 'bin'`
	`find ./ -iname 'lib'`
	`find ./ -iname 'include'`
	`find ./ -iname 'Testing'`
	-rf)    