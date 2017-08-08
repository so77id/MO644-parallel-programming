project(T2 C)

find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_executable(build/t2.bin main.c)
target_link_libraries(build/t2.bin)