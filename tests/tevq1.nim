
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
  link(self(), evq)
  
  #let id = hatch ticker(evq)

  echo "sleep"
  evq.sleep(1)
  echo "slept"

  while true:
    var buf = newString(1024)
    let r = evq.read(0, buf[0].addr, 1024)
    if r <= 0:
      break
    buf.setLen r
    echo "=== ", buf
    if buf.contains("boom"):
      raise newException(IOError, "flap")



proc go() =
  var pool = newPool(4)
  discard pool.hatch main2()
  pool.join()

go()


