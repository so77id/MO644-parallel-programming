project(T1 C)

find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_executable(build/t1.bin main.c)
target_link_libraries(build/t1.bin)