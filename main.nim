
import cps
import actors
import times
import os
import strformat

type

  MsgQuestion = ref object of Message
    a, b: int

  MsgAnswer = ref object of Message
    c: int

  MsgStop = ref object of Message
  
  MsgHello = ref object of Message

  MsgSleep = ref object of Message



# This thing calculates things, but quite slowly

proc calculator() {.cps:Actor.} =

  while true:
    let m = recv()

    if m of MsgQuestion:
      echo &"calculator got a question from {m.src}"
      let mq = m.MsgQuestion
      os.sleep(10)
      send(m.src, MsgAnswer(c: mq.a + mq.b))

    if m of MsgStop:
      break
      
  echo "calculator is done"


proc bob(idCalculator: ActorId, count: int) {.cps:Actor.} =

  var i = 0

  while i < count:
   
    os.sleep(1)
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


proc claire(count: int) {.cps:Actor.} =

  let self = getMyId()

  var i = 0
  while i < count:
    send(self, MsgHello())
    discard recv()
    spin(1e-6)
    i = i + 1


proc sleepy() {.cps:Actor.} = 
  os.sleep(10)


proc main() {.cps:Actor.} =

  # Hatch a calculator

  let idCalculator = hatch calculator()
  
  var bobs = 0

  # Hatch a number of bobs

  for i in 1..20:
    bobs.inc
    discard hatch bob(idCalculator, 20)

  # Wait for all the bobs to finish, then kill the calculator

  while true:

    let m = recv()

    if m of MessageDied:
      let md = m.MessageDied
      bobs.dec
      echo &"actor {md.id} died, {bobs} bobs left!"
      if bobs == 0:
        send(idCalculator, MsgStop())
        break

  echo "main is done"


proc go() =
  var pool = newPool(4)
  discard pool.hatch main()
  pool.run()


go()

