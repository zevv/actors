
import os
import actor


type

  MyMsg = ref object of Message
    a, b: int


proc receiver(a: Actor) {.nimcall, thread, gcsafe.} =
  for i in 1..3:
    echo "pre recv"
    let m = a.recv()
    if m is MyMsg:
      echo m.MyMsg.val


proc sender(a: Actor) {.nimcall, thread, gcsafe.} =
  for i in 1..3:
    let m = MyMsg(val: 32)
    send("receiver", m)
    sleep(100)



spawn("receiver", receiver)
spawn("sender", sender)

joinAll()

