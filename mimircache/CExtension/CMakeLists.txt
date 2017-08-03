# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

# 项目信息
project (mimircache)
set (PROJECT_WEB "http://mimircache.info") 
set (akamaiSimulator_VERSION_MAJOR 0)
set (akamaiSimulator_VERSION_MINOR 1)
set (akamaiSimulator_VERSION_REVISION 0)



# add_compile_options(-std=c++11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")


include_directories(akamaiSimulator/include)
include_directories(akamaiSimulator/libketama/include)
include_directories(akamaiSimulator/utils/include)
include_directories(headers)
include_directories(cache/include)
include_directories(cacheReader/include)
include_directories(profiler/include)
include_directories(utils/include)




set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")



SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 



find_package(GLib "2.40" REQUIRED)
include_directories(${GLib_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GLib_LIBRARY}) 


message(STATUS "gLib found? " ${GLib_FOUND})
message(STATUS "LIBS = " ${LIBS})
message(STATUS "GLib header = " ${GLib_INCLUDE_DIRS})



# 查找目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(akamaiSimulator dirAkamaiSimulator)
aux_source_directory(. DIR_SRCS)

message(STATUS "akamaiSimulator sources = " ${dirAkamaiSimulator}) 


# 添加子目录
add_subdirectory(cache)
add_subdirectory(cacheReader)
add_subdirectory(utils)
add_subdirectory(profiler)

add_subdirectory(akamaiSimulator/libketama) 

# 指定生成目标
add_executable(mimircache ${DIR_SRCS})
add_executable(akamaiSimulatorBin ${dirAkamaiSimulator})

# 添加链接库
target_link_libraries(mimircache cache cacheReader profiler utils m ${LIBS} ${CMAKE_THREAD_LIBS_INIT} -lpthread)
target_link_libraries(akamaiSimulatorBin libketama cache cacheReader profiler utils m ${LIBS} ${CMAKE_THREAD_LIBS_INIT} -lpthread)





# mkdir build
# cd    build
# cmake -G Ninja ..
# ninja             # Parallel build (no need -j12)


# rm -r *; cmake .. -G Ninja; ninja 
# cmake ..; make -j12 


# cmake -DCMAKE_BUILD_TYPE=Debug .. 
# cmake -DCMAKE_BUILD_TYPE=Release .. 