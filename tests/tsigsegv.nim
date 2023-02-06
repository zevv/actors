
import std/monotimes
import actors
import valgrind


type

  Foo = ref object
    val: int

  MessageReq = ref object of Message
    val: int

  MessageRsp = ref object of Message
    val: int


proc alice() {.actor.} =
  while true:
    var req = recv(MessageReq)
    var rsp = MessageRsp(val: req.val * req.val)
    req.src.send(rsp)
    req = nil # TODO: cps keeps this in env
    rsp = nil # TODO: cps keeps this in env


proc bob(alice: Actor) {.actor.} =

  var i = 0
  let n = 10

  while i < n:
    var req = MessageReq(val: i)
    alice.send(req)
    req = nil # TODO: cps keeps this in env
    var rsp = recv().MessageRsp
    rsp = nil # TODO: cps keeps this in env
    inc i

  kill alice



proc main() =
  let pool = newPool(4)
  let a = pool.hatch alice()
  let b = pool.hatch bob(a)
  pool.join()


for i in 1..1000:
  main()
