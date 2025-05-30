# Core dependency management for the framework
# Handles external library integration and versioning

set(ENV{CPM_SOURCE_CACHE} "${PROJECT_SOURCE_DIR}/.cpmcache")

############################################################################################################################
# Boost
############################################################################################################################

# Boost configuration
# Required for utility functions and networking
set(BOOST_VERSION 1.76.0)
set(BOOST_COMPONENTS system filesystem)
include(${PROJECT_SOURCE_DIR}/cmake/fetch_boost.cmake)
fetch_boost_library(core)
fetch_boost_library(smart_ptr)
fetch_boost_library(container)

############################################################################################################################
# yaml-cpp
############################################################################################################################

CPMAddPackage(
  NAME yaml-cpp
  GITHUB_REPOSITORY jbeder/yaml-cpp
  GIT_TAG 0.8.0
  OPTIONS
    "YAML_CPP_BUILD_TESTS OFF"
    "YAML_CPP_BUILD_TOOLS OFF"
    "YAML_BUILD_SHARED_LIBS OFF"
)


############################################################################################################################
# googletest
############################################################################################################################

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.13.0
  VERSION 1.13.0
  OPTIONS "INSTALL_GTEST OFF"
)

############################################################################################################################
# boost-ext reflect : https://github.com/boost-ext/reflect
############################################################################################################################

CPMAddPackage(
  NAME reflect
  GITHUB_REPOSITORY boost-ext/reflect
  GIT_TAG v1.1.1
)

############################################################################################################################
# fmt : https://github.com/fmtlib/fmt
############################################################################################################################

# CLI11 for command line parsing
# Provides intuitive interface for parameter handling
set(CLI11_VERSION 2.2.0)

CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.0.1
)

############################################################################################################################
# magic_enum : https://github.com/Neargye/magic_enum
############################################################################################################################

CPMAddPackage(
  NAME magic_enum
  GITHUB_REPOSITORY Neargye/magic_enum
  GIT_TAG v0.9.6
)

CPMAddPackage(
  NAME xtl
  GITHUB_REPOSITORY xtensor-stack/xtl
  GIT_TAG 0.7.7
  OPTIONS
    "XTL_ENABLE_TESTS Off"
)

CPMAddPackage(
  NAME xtensor
  GITHUB_REPOSITORY xtensor-stack/xtensor
  GIT_TAG 0.25.0
  OPTIONS
    "XTENSOR_ENABLE_TESTS Off"
)

# MessagePack for serialization
# Efficient binary format for model storage
set(MSGPACK_VERSION 3.3.0)
include(${PROJECT_SOURCE_DIR}/cmake/fetch_msgpack.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/fetch_cli11.cmake)