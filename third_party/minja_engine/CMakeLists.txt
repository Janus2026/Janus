include(cc_library)

cc_library(
  NAME 
    minja
  SRCS 
    include/minja/minja.hpp
    include/minja/chat-template.hpp
  INCLUDES 
    include
  DEPS
    nlohmann_json::nlohmann_json
)
