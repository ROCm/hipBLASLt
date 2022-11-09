********************************
Library Source Code Organization
********************************

The hipBLASLt code is split into two major parts:

- The `library` directory contains all source code for the library.
- The `clients` directory contains all test code and code to build clients.
- Infrastructure

The `library` directory
-----------------------

library/include
```````````````
Contains C98 include files for the external API. These files also contain Doxygen
comments that document the API.

library/src/amd_detail
```````````````````````
Implementation of hipBLASLt interface compatible with rocBLASLt APIs.

library/src/include
```````````````````
Internal include files for:

- Converting C++ exceptions to hipBLAS status.

The `clients` directory
-----------------------

clients/samples
```````````````
Sample code for calling hipBLASLt functions.


Infrastructure
--------------

- CMake is used to build and package hipBLASLt. There are CMakeLists.txt files throughout the code.
- Doxygen/Breathe/Sphinx/ReadTheDocs are used to produce documentation. Content for the documentation is from:

  - Doxygen comments in include files in the directory library/include
  - files in the directory docs/source.

- Jenkins is used to automate Continuous Integration testing.
- clang-format is used to format C++ code.


