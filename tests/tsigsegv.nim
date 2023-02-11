
import std/monotimes
import actors
import valgrind

type
  Ping = ref object of Message

proc alice() {.actor.} =
  alice.send(Ping())
  var req = recv()

proc main() =
  let pool = newPool(4)
  let a = pool.hatch alice()
  pool.join()

main()
