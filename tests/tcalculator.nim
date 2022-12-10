
import std/os
import std/syncio
import std/strformat
import std/strutils
import std/times
import std/posix
import std/atomics


import nimactors


type

  MsgQuestion = ref object of Message
    a, b: int

  MsgAnswer = ref object of Message
    c: int

  MsgStop = ref object of Message


# This thing calculates things

proc calculator() {.actor.} =

  receive:
    (src, a, b) = MsgQuestion(src: src, a:a, b:b):
      send(src, MsgAnswer(c: a + b))

    MsgStop:
      break
      
  echo "calculator is done"


# This is Bob. Bob asks questions to the calculator

proc bob(calc: Actor, count: int) {.actor.} =
  var i = 0
  while i < count:
    send(calc, MsgQuestion(a: 10, b: i))
    let m = recv()
    if m of MsgAnswer:
      let ma = m.MsgAnswer
    inc i


proc main() {.actor.} =

  # Create one calculator and a large number of bobs to give it work.
 
  var kids = 0
  
  let calc = hatch calculator()
  
  var i = 0
  while i < 1500:
    let i2 = hatch bob(calc, 50)
    inc kids
    inc i

  # Wait for all the kids to finish, then kill the calculator

  receive:
    MessageExit():
      kids.dec
      if kids == 0:
        send(calc, MsgStop())
        break

  echo "main is done"


proc go() =
  var pool = newPool(4)
  discard pool.hatch main()
  pool.join()


go()


