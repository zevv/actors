
import std/monotimes
import actors
import actors/valgrind


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


proc bob(alice: Actor) {.actor.} =

  var i = 0

  let t1 = getMonoTime().ticks.float / 1.0e9

  var n = 100_000
  if running_on_valgrind():
    n = 1000

  while i < n:
    var req = MessageReq(val: i)
    alice.send(req)
    let rsp = recv().MessageRsp
    if (i mod 100_000) == 0: echo rsp.val
    inc i

  let t2 = getMonoTime().ticks.float / 1.0e9

  let msg_s = (i.float / (t2-t1)).int

  echo msg_s /% 1000, " Kcalls/sec"

  kill alice



proc main() =
  let pool = newPool(4)
  let a = pool.hatch alice()
  let b = pool.hatch bob(a)
  pool.join()


for i in 1..4:
  main()
