
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


proc sendself() {.cps:Actor.} =
  let self = getMyId()
  echo "sending"
  send(self, MsgSleep())
  echo "prerecv"
  discard recv()
  echo "postrev"


# This thing answers questions

proc alice() {.cps:Actor.} =

  while true:
    let m = recv()

    if m of MsgQuestion:
      echo &"alice got a question from {m.src}"
      let mq = m.MsgQuestion
      os.sleep(10)
      send(m.src, MsgAnswer(c: mq.a + mq.b))

    if m of MsgStop:
      break
      
  echo "alice is done"


proc bob(idAlice: ActorId, count: int) {.cps:Actor.} =

  sendself()

  var i = 0

  while i < count:
    # Let's ask alice a question
    
    send(idAlice, MsgQuestion(a: 10, b: i))

    # And receive the answer
    let m = recv()

    if m of MsgAnswer:
      let ma = m.MsgAnswer
      echo &"bob received an answer from {ma.src}: {ma.c}"

    inc i

  # Thank you alice, you can go now

  send(idAlice, MsgStop())
  echo "bob is done"


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
    #sleep(1)
    spin(1e-6)
    i = i + 1


proc main() =

  var pool = newPool(16)


  for i in 1..100:
    let idAlice = pool.hatch alice()
    let idBob = pool.hatch bob(idAlice, 10)

  #for i in 1..10:
  #  let idClaire = pool.hatch claire(1000)

  pool.run()


main()

