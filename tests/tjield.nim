
import std/os

import nimactors

proc thing(i: int) {.actor.} =
  echo "start thing"
  while true:
    echo i
    os.sleep(10)
    jield()

proc main() {.actor.} =
  let pid1 = hatch thing(1)
  let pid2 = hatch thing(2)
  os.sleep(50)
  echo "killing"
  kill pid1
  kill pid2

var pool = newPool(2)
discard pool.hatch main()
pool.join()

