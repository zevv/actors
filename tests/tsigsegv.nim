
import std/monotimes
import actors
import valgrind


type

  Foo = ref object
    val: int

  Ping = ref object of Message
    val: int

  Pong = ref object of Message
    val: int


proc alice() {.actor.} =
  while true:
    var req = recv(Ping)
    var rsp = Pong(val: req.val * req.val)
    req.src.send(rsp)

    req = nil
    rsp = nil


proc bob(alice: Actor) {.actor.} =

  var req = Ping(val: 24)
  alice.send(req)
  var rsp = recv().Pong

  req = nil
  rsp = nil

  kill alice



proc main() =
  let pool = newPool(4)
  let a = pool.hatch alice()
  let b = pool.hatch bob(a)
  pool.join()


for i in 1..1000:
  main()
