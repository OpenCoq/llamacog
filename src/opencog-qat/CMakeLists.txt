cmake_minimum_required(VERSION 3.14)

# OpenCog QAT Framework Library
add_library(opencog_qat
    synthetic-calibration.cpp
    opencog-quantization.cpp
    qat-framework.cpp
)

target_include_directories(opencog_qat PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

target_link_libraries(opencog_qat PRIVATE
    llama
    common
)

# Set C++ standard
target_compile_features(opencog_qat PRIVATE cxx_std_17)

# OpenCog QAT Command Line Tool
if(LLAMA_BUILD_TOOLS)
    add_executable(llama-opencog-qat ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/opencog-qat/opencog-qat.cpp)
    
    target_link_libraries(llama-opencog-qat PRIVATE
        opencog_qat
        llama
        common
    )
    
    target_include_directories(llama-opencog-qat PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    )
    
    target_compile_features(llama-opencog-qat PRIVATE cxx_std_17)
endif()

# Add OpenCog QAT tests if testing is enabled
if(LLAMA_BUILD_TESTS)
    add_executable(test-opencog-qat-framework ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/test-opencog-qat.cpp)
    
    target_link_libraries(test-opencog-qat-framework PRIVATE
        opencog_qat
        llama
        common
    )
    
    target_include_directories(test-opencog-qat-framework PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    )
    
    target_compile_features(test-opencog-qat-framework PRIVATE cxx_std_17)
endif()