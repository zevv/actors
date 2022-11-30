
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
  var i = 0
  while i < n:
    discard hatch main4d(n)
    inc i

proc main4b(n: int) {.actor.} =
  var i = 0
  while i < n:
    discard hatch main4c(n)
    inc i
    
proc main4a(n: int) {.actor.} =
  var i = 0
  while i < n:
    discard hatch main4b(n)
    inc i

proc main4(n: int) {.actor.} =
  var i = 0
  while i < n:
    stderr.write(".")
    discard hatch main4a(n)
    inc i


proc go() =
  let n = 50
  var pool = newPool(4)
  discard pool.hatch main4(n)
  pool.join()
  doAssert ntotal.load() == n ^ 4
  echo ""
  echo "all good, n = ", ntotal.load()

go()


