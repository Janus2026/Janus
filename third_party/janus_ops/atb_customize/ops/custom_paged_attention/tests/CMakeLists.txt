set(CMAKE_SKIP_RPATH FALSE)

link_directories("${ATB_HOME_PATH}/lib")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

add_executable(custom_paged_attention_test 
    custom_paged_attention_test.cpp
)
target_link_libraries(custom_paged_attention_test
  PRIVATE
    atb_customize
    atb
    pthread
    GTest::gtest_main
)

set_target_properties(custom_paged_attention_test PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:atb_customize>"
)

set_target_properties(custom_paged_attention_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/output/bin")

include(GoogleTest)
add_test(NAME custom_paged_attention_test
         COMMAND custom_paged_attention_test)
