
import std/os

import nimactors

proc unkillable() {.actor.} =
  while true:
    let m = tryRecv()
    echo "killme ", m
    os.sleep(100)
    jield()


proc main() {.actor.} =
  let pid = hatch unkillable()
  os.sleep(500)
  kill pid


var pool = newPool(4)
discard pool.hatch main()
pool.join()

