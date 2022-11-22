
import cps
import actors
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


proc sendself() {.cps:Work.} =
  echo "sending"
  send("bob", MsgSleep())
  echo "prerecv"
  discard recv()
  echo "postrev"


# This thing answers questions

proc alice() {.cps:Work.} =
  echo "I am alice"

  while true:
    let m = recv()

    if m of MsgQuestion:
      echo &"alice got a question from {m.src}"
      let mq = m.MsgQuestion
      send("bob", MsgAnswer(c: mq.a + mq.b))

    if m of MsgStop:
      echo "alice says bye"
      break


proc bob() {.cps:Work.} =

  echo "I am bob"

  sendself()

  var i = 0

  while i < 5:
    # Let's ask alice a question
    
    send("alice", MsgQuestion(a: 10, b: i))

    # And receive the answer
    let m = recv()

    if m of MsgAnswer:
      let ma = m.MsgAnswer
      echo &"bob received an answer from {ma.src}: {ma.c}"

    inc i

  # Thank you alice, you can go now

  send("alice", MsgStop())


proc claire(count: int) {.cps:Work.} =

  var i = 0
  while i < count:
    send("claire", MsgHello())
    discard recv()
    os.sleep(100)
    i = i + 1


proc main() =

  var pool = newPool(1)

  pool.hatch "alice", alice()
  pool.hatch "bob", bob()
  pool.hatch "claire", claire(3)

  pool.run()


main()

