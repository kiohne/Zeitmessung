cmake_minimum_required(VERSION "3.17.3")
project(test_project)

set( NAME_SRC
    Quelle.cpp    
)
set( NAME_HEADERS       
	 include/opencv2/core.hpp
	 include/opencv2/highgui.hpp
	 include/opencv2/imgproc.hpp
	 include/opencv2/core/cvstd.hpp
	 include/opencv2/core/cuda.hpp
	 include/opencv2/cudawarping.hpp
	 include/opencv2/cudaarithm.hpp
	 include/opencv2/cudaimgproc.hpp
	 include/opencv2/cudafilters.hpp
)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/include/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(test ${NAME_SRC} ${NAME_HEADERS})

link_libraries(${CMAKE_SOURCE_DIR}/include/lib)
target_link_libraries(test opencv_world450.lib)
