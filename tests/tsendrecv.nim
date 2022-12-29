
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
    let req = recv(MessageReq)
    let rsp = MessageRsp(val: req.val * req.val)
    req.src.sendCps(rsp)


proc bob(alice: Actor) {.actor.} =

  var i = 0

  let t1 = getMonoTime().ticks.float / 1.0e9

  var n = 1_000_000
  if running_on_valgrind():
    n = 100_000

  while i < n:
    let req = MessageReq(val: i)
    alice.sendCps(req)
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


main()
