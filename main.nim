
import os
import actor


type

  MQuestion = ref object of Message
    a, b: int
  
  MAnswer = ref object of Message
    sum: int


proc receiver(a: Actor) {.nimcall, thread, gcsafe.} =
  let m = a.recv()
  let sum = m.MQuestion.a + m.MQuestion.b
  a.send(m.src, Manswer(sum: sum))


proc sender(a: Actor) {.nimcall, thread, gcsafe.} =
  let m = MQuestion(a: 10, b:5)
  a.send("receiver", m)
  let ma = a.recv()
  echo ma.MAnswer.sum




spawn("receiver", receiver)
spawn("sender", sender)

joinAll()

