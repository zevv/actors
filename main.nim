
import cps
import actors
import times
import os
import strformat
import isisolated


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

  let self = getMyId()

  var i = 0
  while i < count:
    send(self, MsgHello())
    discard recv()
    i = i + 1


proc sleepy() {.actor.} = 
  os.sleep(10)


proc main() {.actor.} =

  var kids = 0
  
  claire(10)

  discard hatch claire(10)
  inc kids

  let idCalculator = hatch calculator()
  
  var i = 0
  while i < 20:
    discard hatch bob(idCalculator, 100)
    inc kids
    inc i

  # Wait for all the kids to finish, then kill the calculator

  while true:

    let m = recv()

    if m of MessageDied:
      let md = m.MessageDied
      kids.dec
      echo &"actor {md.id} died, {kids} kids left!"
      if kids == 0:
        send(idCalculator, MsgStop())
        break

  echo "main is done"


proc go() =
  var pool = newPool(4)
  discard pool.hatch main()
  pool.run()


go()


