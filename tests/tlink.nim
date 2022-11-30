
import std/os

import nimactors

proc other() {.actor.} =
  while true:
    echo "tick"
    os.sleep(100)
    jield()


proc main() {.actor.} =
  let pid = hatchLinked other()
  os.sleep(500)


var pool = newPool(4)
discard pool.hatch main()
pool.join()

