

import std/macros
import std/os

import actors

type
  Message1 = ref object of Message
    val: int
    weight: float

  Message2 = ref object of Message
    name: string

  Message3 = ref object of Message
    thing: float
  
  Message4 = ref object of Message


proc sender(dst: Actor) {.actor.} =
  send(dst, Message4())
  send(dst, Message1(val: 122, weight: 14.4))
  send(dst, Message3(thing: 3.14))
  send(dst, Message4())
  send(dst, Message2(name: "charlie"))
  send(dst, Message1(val: 124))
  send(dst, Message1(val: 123))
  send(dst, Message1(val: 123))
  send(dst, Message4())


proc main() {.actor.} =

  discard hatch sender(self())

  var n = 0

  block:
    receive:

      Message1(val: 123):
        echo "got Message1, val was a direct hit 123"
        inc n
        doAssert n == 5 or n == 6

      (v, w) = Message1(val: v, weight: w):
        echo "got Message1, val was ", v, " weight ", w
        inc n
        doassert n == 1 or n == 4

      name = Message2(name: name):
        echo "got Message2, name was ", name
        inc n
        doAssert n == 3

      Message3():
        echo "Got Message3"
        inc n
        doAssert n == 2
      
      r = MessageExit(reason: r):
        echo "Got MessageExit, reason: ", r
        break

  discard recv(Message4)
  discard recv(Message4)
  discard recv(Message4)

  echo "all good"


var pool = newPool(4)
discard pool.hatch main()
pool.join()
