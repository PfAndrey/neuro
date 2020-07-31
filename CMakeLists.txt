cmake_minimum_required(VERSION 3.5)
project(Neuro VERSION 0.99 LANGUAGES C CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
 
if(NOT CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE "Release")
endif()

set(SOURCE
	${CMAKE_SOURCE_DIR}/source/main.cpp
	${CMAKE_SOURCE_DIR}/source/Matrix.h
	${CMAKE_SOURCE_DIR}/source/NeuralNetwork.h
	${CMAKE_SOURCE_DIR}/source/NeuralNetwork.cpp
	${CMAKE_SOURCE_DIR}/source/ActivationFunctions.h)


# ADD EXECUTABLE
add_executable(Neuro ${SOURCE})
if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Neuro)
	set_target_properties( Neuro PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/Build" CXX_STANDARD 17)
	set_target_properties( Neuro PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/Build" CXX_STANDARD 17)
#	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif(MSVC)

 