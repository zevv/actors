
import std/os

import nimactors
import nimactors/mallinfo

proc other() {.actor.} =
  discard recv()

proc main() {.actor.} =

  var i = 0
  var mtot = 0.uint

  while i < 100:
    let m1 = mallinfo2().uordblks
    let pid = hatch other()
    let m2 = mallinfo2().uordblks
    os.sleep(1)
    kill pid
    mtot += (m2 - m1)
    inc i

  let usage = mtot.int / i

  echo "memore usage per actor: ", usage, " bytes"

  when defined(danger):
    doAssert usage < 796

  when defined(release):
    doAssert usage < 796

  os.sleep(100)
  echo "all good"


var pool = newPool(4)
discard pool.hatch main()
pool.join()

