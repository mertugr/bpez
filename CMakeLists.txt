cmake_minimum_required(VERSION 3.14)
project(bpez VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(bpez_lib src/bpe.cpp)
target_include_directories(bpez_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(bpez::bpez ALIAS bpez_lib)

add_executable(bpez src/main.cpp)
target_link_libraries(bpez PRIVATE bpez_lib)

enable_testing()
add_executable(bpez_tests tests/test_bpe.cpp)
target_link_libraries(bpez_tests PRIVATE bpez_lib)
add_test(NAME bpez_tests COMMAND bpez_tests)

install(TARGETS bpez bpez_lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/bpez DESTINATION include)
