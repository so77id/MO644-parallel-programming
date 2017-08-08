project(Tests C)

find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_executable(build/hello.bin hello.c)
target_link_libraries(build/hello.bin)

add_executable(build/hello2.bin hello2.c)
target_link_libraries(build/hello2.bin)

add_executable(build/sum_array.bin sum_array.c)
target_link_libraries(build/sum_array.bin)

add_executable(build/sum_array2.bin sum_array2.c)
target_link_libraries(build/sum_array2.bin)