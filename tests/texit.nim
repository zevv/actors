
import std/os

import actors

proc alice() {.actor.} =
  while true:
    discard recv()


proc bob() {.actor.} =
  discard


proc claire() {.actor.} =
  raise newException(IOError, "flap")


proc main() {.actor.} =

  block:
    let pid = hatch alice()
    os.sleep(5)
    kill pid
    let m = recv(MessageExit)
    doAssert m.reason == Killed

  block:
    let pid = hatch bob()
    let m = recv(MessageExit)
    doAssert m.reason == Normal
  
  block:
    let pid = hatch claire()
    let m = recv(MessageExit)
    doAssert m.reason == Error
    doAssert not m.ex.isNil
    doAssert m.ex.msg == "flap"

  echo "all good"

var pool = newPool(4)
discard pool.hatch main()
pool.join()

