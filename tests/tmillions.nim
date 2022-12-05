
when not defined(release):
  quit 0 # too slow

import std/os
import std/syncio
import std/math
import std/strformat
import std/strutils
import std/times
import std/posix
import std/atomics

import nimactors

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
  let n = 50
  var pool = newPool(16)
  discard pool.hatch main4(n)
  echo "hatched"

  while true:
    let n = ntotal.load()
    echo n
    if n == 6_377_551:
      break
    os.sleep(250)

  pool.join()
  echo "all good"

go()


