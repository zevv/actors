
# nim c --verbosity:0 --mm:arc --panics:on -d:danger tests/tboom
# valgrind --exit-on-first-error=yes --error-exitcode=255 --quiet --tool=helgrind tests/tboom

import std/os

import actors

proc bob() {.actor.} =
  discard

proc workwork() {.actor.} =
  discard hatch bob()
  discard recv(MessageExit)

for i in 1..100:
  let pool = newPool(4)
  discard pool.hatch workwork()
  pool.join()

