
import os
import actors


type

  MQuestion = ref object of Message
    a, b: int
  
  MAnswer = ref object of Message
    sum: int

  MStop = ref object of Message


proc receiver(a: Actor) {.nimcall, thread, gcsafe.} =

  while true:

    let m = a.recv()

    if m of MQuestion:
      echo "receiver: got question"
      let sum = m.MQuestion.a + m.MQuestion.b
      a.send(m.src, Manswer(sum: sum))

    if m of MStop:
      echo "receiver: got stop"
      break


proc sender(a: Actor) {.nimcall, thread, gcsafe.} =

  a.send("receiver", MQuestion(a: 10, b:5))

  let ma = a.recv()

  if ma of MAnswer:
    echo "sender: got answer"
    echo ma.MAnswer.sum

  echo "Sendstop"
  a.send("receiver", MStop())


spawn("receiver", receiver)
spawn("sender", sender)

joinAll()
