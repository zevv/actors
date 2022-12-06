
import std/monotimes
import nimactors


type

  Foo = ref object
    val: int

  MessageHello = ref object of Message
    foo: Foo


proc alice() {.actor.} =
  while true:
    let m = recv()
    m.src.send(MessageHello())


proc bob(alice: Actor) {.actor.} =
  var i = 0

  let t1 = getMonoTime().ticks.float / 1.0e9
  while i < 1_000_000:
    alice.send(MessageHello())
    discard recv()
    inc i
  let t2 = getMonoTime().ticks.float / 1.0e9

  let msg_s = (i.float / (t2-t1)).int

  echo msg_s /% 1000, " Kcalls/sec"

  kill alice



proc main() =
  let pool = newPool(2)
  let a = pool.hatch alice()
  let b = pool.hatch bob(a)
  pool.join()


main()
