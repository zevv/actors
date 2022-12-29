
import std/os

import actors

proc other() {.actor.} =
  while true:
    echo "tick"
    os.sleep(100)
    jield()


proc main() {.actor.} =
  let pid = hatchLinked other()
  os.sleep(500)
  echo "bye"


var pool = newPool(4)
discard pool.hatch main()
pool.join()

