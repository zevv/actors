
import std/os
import std/strformat
import std/strutils
import std/times
import std/posix


import nimactors
import nimactors/lib/evq


type

  MsgQuestion = ref object of Message
    a, b: int

  MsgAnswer = ref object of Message
    c: int

  MsgStop = ref object of Message
  
  MsgHello = ref object of Message

  MsgSleep = ref object of Message



# This thing calculates things, but quite slowly

proc calculator() {.actor.} =

  while true:
    os.sleep(10)
    let m = recv()

    if m of MsgQuestion:
      echo &"calculator got a question from {m.src}"
      let mq = m.MsgQuestion
      send(m.src, MsgAnswer(c: mq.a + mq.b))

    if m of MsgStop:
      break
      
  echo "calculator is done"


proc bob(idCalculator: ActorId, count: int) {.actor.} =

  var i = 0

  while i < count:
   
    send(idCalculator, MsgQuestion(a: 10, b: i))

    let m = recv()

    if m of MsgAnswer:
      let ma = m.MsgAnswer
      echo &"bob received an answer from {ma.src}: {ma.c}"

    inc i



proc spin(t: float) =
  let t_done = epochTime() + t
  while epochTime() < t_done:
    discard


proc claire(count: int) {.actor.} =

  var i = 0
  let me = self()
  while i < count:
    send(me, MsgHello())
    discard recv()
    i = i + 1



proc main() {.actor.} =

  var kids = 0
  
  #claire(10)

  let i1 = hatch claire(10)
  inc kids

  let idCalculator = hatch calculator()
  
  var i = 0
  while i < 10:
    let i2 = hatch bob(idCalculator, 2)
    inc kids
    inc i

  # Wait for all the kids to finish, then kill the calculator

  while true:

    let md = recv(MessageExit)
    kids.dec
    echo &"actor {md.id} died, reason: {md.reason}, {kids} kids left!"
    if md.reason == erError:
      echo "An exception occured: ", md.ex.msg, "\n", md.ex.getStackTrace()
    if kids == 0:
      send(idCalculator, MsgStop())
      break

  echo "recv mesg"
  let m = recv()
  echo "Got mesg"

  echo "main is done"



proc ticker() {.actor.} = 
  while true:
    echo "-----------------------"
    os.sleep(100)


proc readFd(evq: Evq, fd: cint): string {.actor.} =
  evq.addFd(fd)
  discard recv(MessageEvqEvent)
  var buf = newString(1024)
  let r = posix.read(0, buf[0].addr, buf.len)
  buf.setLen if r > 0: r else: 0
  evq.delFd(fd)
  buf


proc main2() {.actor.} =
  
  let evq = newEvq()

  echo "sleep"
  evq.sleep(1)
  echo "slept"

  #let id = hatch ticker()

  while true:
    let l = readFd(evq, 0)
    if l.len == 0:
      break
    echo "=== ", l
    if l.contains("boom"):
      raise newException(IOError, "flap")

  echo "main2 done, killing ", evq.id
  kill(evq.id)


proc go() =
  var pool = newPool(2)

  #discard pool.hatch main()
  discard pool.hatch main2()

  pool.join()


go()


