
import std/os
import std/syncio
import std/math
import std/strformat
import std/strutils
import std/times
import std/posix
import std/atomics

import actors
import valgrind

var ntotal: Atomic[int]

proc main4d(n: int) {.actor.} =
  ntotal += 1
  discard

proc main4c(n: int) {.actor.} =
  ntotal += 1
  var i = 0
  while i < n:
    discard hatch main4d(n)
    inc i

proc main4b(n: int) {.actor.} =
  ntotal += 1
  var i = 0
  while i < n:
    discard hatch main4c(n)
    inc i
    
proc main4a(n: int) {.actor.} =
  ntotal += 1
  var i = 0
  while i < n:
    discard hatch main4b(n)
    inc i

proc main4(n: int) {.actor.} =
  ntotal += 1
  var i = 0
  while i < n:
    stderr.write(".")
    discard hatch main4a(n)
    inc i


proc go() =

  var count = 50
  when not defined(release):
    count = 20
  if running_on_valgrind():
    echo "valgrind detected"
    count = 7

  var pool = newPool(16)
  discard pool.hatch main4(count)
  echo "hatched"

  let total = (count ^ 0) + (count ^ 1) + (count ^ 2) + (count ^ 3) + (count ^ 4)
  echo total

  while true:
    let n = ntotal.load()
    echo n
    if n == total:
      break
    os.sleep(250)

  pool.join()
  echo "all good"

go()


