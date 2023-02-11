
import std/monotimes
import actors
import valgrind

type
  Ping = ref object of Message

proc alice() {.actor.} =
  var req = recv()

proc bob(alice: Actor) {.actor.} =
  alice.send(Ping())

proc main() =
  let pool = newPool(4)
  let a = pool.hatch alice()
  let b = pool.hatch bob(a)
  pool.join()

main()
