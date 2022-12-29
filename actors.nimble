version = "0.0.1"
author = "Zevv"
description = "actors"
license = "MIT"

requires "https://github.com/nim-works/cps >= 0.6.0 & < 1.0.0"
requires "https://github.com/disruptek/balls >= 3.9.5 & < 4.0.0"


task test, "run tests for ci":
  exec "balls --gc:arc -d:useMalloc"
