set(CMAKE_SKIP_RPATH FALSE)

link_directories("${ATB_HOME_PATH}/lib")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

add_executable(customize_blockcopy_test customize_blockcopy_test.cpp)
target_link_libraries(customize_blockcopy_test
  PRIVATE
    atb_customize
    atb
    pthread
    GTest::gtest_main
)

set_target_properties(customize_blockcopy_test PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:atb_customize>"
)

set_target_properties(customize_blockcopy_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/output/bin")

include(GoogleTest)
add_test(NAME customize_blockcopy_test
         COMMAND customize_blockcopy_test)