
import std/monotimes
import actors
import valgrind


type

  Ping = ref object of Message

  Pong = ref object of Message


proc alice() {.actor.} =
  while true:
    var req = recv(Ping)
    var rsp = Pong()
    req.src.send(rsp)


proc bob(alice: Actor) {.actor.} =

  var req = Ping()
  alice.send(req)
  var rsp = recv().Pong

  kill alice



proc main() =
  let pool = newPool(4)
  let a = pool.hatch alice()
  let b = pool.hatch bob(a)
  pool.join()


for i in 1..1000:
  main()
