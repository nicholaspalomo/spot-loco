cmake_minimum_required(VERSION 3.10)
project(_raisim_gym)
set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 2.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(raisimOgre 0.6.0 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -fopenmp -w -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

message("[RAISIM_GYM] prefix path ${CMAKE_PREFIX_PATH}")
message("[RAISIM_GYM] including directory ${RSG_ENVIRONMENT_INCLUDE_PATH}")

if (TRAIN_STUDENT)
  add_definitions(-DTRAIN_STUDENT)
endif()

if(CUSTOM_ENVIRONMENT)

    message(STATUS "BUILDING CUSTOM ENVIRONMENT")
    message(STATUS "RSG_ENVIRONMENT_INCLUDE_PATH=${RSG_ENVIRONMENT_INCLUDE_PATH}")
    message(STATUS "ENVIRONMENT_BUILD_NAME=${ENVIRONMENT_BUILD_NAME}")

    set(RSG_ENVIRONMENT_INCLUDE_PATH ${ENVIRONMENT_PATH})
    set(RSG_ENVIRONMENT_NAME ${ENVIRONMENT_BUILD_NAME})
    set(SOURCE_DIR ${RSG_ENVIRONMENT_INCLUDE_PATH}/raisim_gym/env/)
    set(INCLUDE_DIR 
        ${RSG_ENVIRONMENT_INCLUDE_PATH} # Contains Environment.hpp and visSetupCallback.hpp
        ${RSG_ENVIRONMENT_INCLUDE_PATH}/raisim_gym/env # Contains VectorizedEnvironment.hpp and others
        ${RSG_ENVIRONMENT_INCLUDE_PATH}/raisim_gym/env/visualizer # Contains  header files for visualizations in raisimOgre
        )

else()

    set(SOURCE_DIR raisim_gym/env)
    set(INCLUDE_DIR raisim_gym/env ${RSG_ENVIRONMENT_INCLUDE_PATH})

endif()

    set(INCLUDE_DIR ${INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})
    pybind11_add_module(_raisim_gym ${SOURCE_DIR}/raisim_gym.cpp)
    target_link_libraries(_raisim_gym PRIVATE raisim::raisim raisim::raisimOgre yaml-cpp)
    target_include_directories(_raisim_gym PUBLIC ${INCLUDE_DIR})
    target_compile_options(_raisim_gym PRIVATE -mtune=native -fPIC -O3)

    message(STATUS "CMAKE INSTALL PREFIX ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "EXECUTABLES INSTALLED TO ${CMAKE_INSTALL_BINDIR}")
    message(STATUS "LIBRARIES INSTALLED TO ${CMAKE_INSTALL_LIBDIR}")

    if (CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("[RAISIM_GYM]BUILDING THE DEBUG APP")
        find_package(PythonLibs REQUIRED)
        
        add_executable(debug_app ${SOURCE_DIR}/debug_app.cpp)
        target_link_libraries(debug_app PRIVATE raisim::raisim raisim::raisimOgre yaml-cpp ${PYTHON_LIBRARIES})
        target_include_directories(debug_app PUBLIC ${SOURCE_DIR} ${RSG_ENVIRONMENT_INCLUDE_PATH} ${EIGEN3_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
        target_compile_options(debug_app PRIVATE -mtune=native -fPIC -g -O0)
    endif()


    # add_executable(envtest
    #        raisim_gym/env/env/ANYmal_on_wheels/test_env.cpp)
    # target_link_libraries(envtest PRIVATE raisim::raisim raisim::raisimOgre yaml-cpp)
    # target_include_directories(envtest PUBLIC ${INCLUDE_DIR})
    # target_compile_options(envtest PRIVATE -mtune=native -fPIC -O3)