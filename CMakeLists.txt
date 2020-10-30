cmake_minimum_required(VERSION 3.0.0)
project(opengl-basic VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 11)

# Add Include Path
include_directories(
	src
	inc
)

#add opengl
find_package(OpenGL)
# add glfw
include_directories(3rdLibs/glfw-3.3.2/include)
link_directories(3rdLibs/lib)
# add glad
include_directories(3rdLibs/glad/include)
add_library(glad 3rdLibs/glad/src/glad.c)
# add imgui
include_directories(3rdLibs/imgui)
# find source codes
# Add source files
set(SOURCE_FILES 
	src/main.cpp
    	src/tiny_obj_loader.cc
	src/tiny_obj_loader.h
	3rdLibs/imgui/imconfig.h
    	3rdLibs/imgui/imgui_demo.cpp
    	3rdLibs/imgui/imgui_draw.cpp
    	3rdLibs/imgui/imgui_impl_glfw.cpp
    	3rdLibs/imgui/imgui_impl_glfw.h
    	3rdLibs/imgui/imgui_impl_opengl3.cpp
    	3rdLibs/imgui/imgui_impl_opengl3.h
    	3rdLibs/imgui/imgui_internal.h
    	3rdLibs/imgui/imgui_widgets.cpp
    	3rdLibs/imgui/imgui.cpp
    	3rdLibs/imgui/imgui.h
    	3rdLibs/imgui/imstb_rectpack.h
    	3rdLibs/imgui/imstb_textedit.h
    	3rdLibs/imgui/imstb_truetype.h
	)
if(WIN32)
add_definitions(-D_AMD64_)
endif()
#message("Following files are complied " ${SOURCE_FILES})
message("Following files are complied " ${SOURCE_FILES})
add_executable(opengl-basic ${SOURCE_FILES})
target_link_libraries(opengl-basic glad glfw3 ${OPENGL_glu_LIBRARY} ${OPENGL_LIBRARIES})

