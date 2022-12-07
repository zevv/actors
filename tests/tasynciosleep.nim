
import std/os
import std/syncio
import std/strformat
import std/strutils
import std/times
import std/posix
import std/atomics


import pkg/nimactors
import pkg/nimactors/lib/evq
import pkg/nimactors/lib/asyncio


proc main2() {.actor.} =
  
  let evq = newEvq()

  echo "sleep"
  evq.sleep(0.1)
  echo "slept"
  evq.sleep(0.1)
  echo "killing evq"
  kill evq


proc go() =
  var pool = newPool(4)
  discard pool.hatch main2()
  pool.join()

go()


