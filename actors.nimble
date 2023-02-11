version = "0.0.1"
author = "Zevv"
description = "actors"
license = "MIT"

requires "https://github.com/nim-works/cps"


task test, "run tests for ci":
  exec "BALLS_VALGRIND_FLAGS='--gen-suppressions=all --suppressions=./misc/valgrind-drd-suppressions' balls --mm:arc -d:useMalloc --path=$projectdir"
