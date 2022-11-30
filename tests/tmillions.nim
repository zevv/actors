
import std/os
import std/syncio
import std/strformat
import std/strutils
import std/times
import std/posix
import std/atomics

import nimactors

proc main4d(n: int) {.actor.} =
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
  echo "a"
  var i = 0
  while i < n:
    discard hatch main4b(n)
    inc i

proc main4(n: int) {.actor.} =
  echo "4"
  var i = 0
  while i < n:
    discard hatch main4a(n)
    inc i


proc go() =
  var pool = newPool(4)
  discard pool.hatch main4(50)
  pool.join()


go()


