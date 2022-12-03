
import std/os
import std/syncio
import std/strformat
import std/strutils
import std/times
import std/posix
import std/atomics


import nimactors
import nimactors/lib/evq


proc main2() {.actor.} =
  
  let evq = newEvq()

  echo "sleep"
  evq.sleep(0.1)
  echo "slept"
  evq.sleep(0.1)
  echo "killing evq"
  os.sleep(1000)
  kill evq


proc go() =
  var pool = newPool(4)
  discard pool.hatch main2()
  pool.join()

go()


