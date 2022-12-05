
import std/os

import nimactors

proc dyer() {.actor.} =
  os.sleep(10)

proc watcher(pid: Actor) {.actor.} =
  monitor(pid)
  doAssert recv() of MessageExit

proc main() {.actor.} =
  # Let one process die while two are watching
  let pid = hatch dyer()
  let w1 = hatchLinked watcher(pid)
  let w2 = hatchLinked watcher(pid)
  os.sleep(20)

var pool = newPool(4)
discard pool.hatch main()
pool.join()

