
import std/os

import nimactors
import nimactors/mallinfo

proc other() {.actor.} =
  discard recv()


proc main() {.actor.} =
    
  let m1 = mallinfo2().uordblks
  discard hatchLinked other()
  discard hatchLinked other()
  discard hatchLinked other()
  discard hatchLinked other()
  discard hatchLinked other()
  discard hatchLinked other()
  discard hatchLinked other()
  discard hatchLinked other()
  discard hatchLinked other()
  discard hatchLinked other()
  let m2 = mallinfo2().uordblks

  let usage = (m2 - m1).int / 10

  echo "memore usage per actor: ", usage, " bytes"

  when defined(danger):
    doAssert usage < 512

  when defined(release):
    doAssert usage < 512

  echo "all good"


var pool = newPool(4)
discard pool.hatch main()
pool.join()

