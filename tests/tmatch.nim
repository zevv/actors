

import std/macros
import std/os

import nimactors

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
  send(dst, Message1(val: 122, weight: 14.4))
  send(dst, Message3(thing: 3.14))
  send(dst, Message2(name: "charlie"))
  send(dst, Message1(val: 124))
  send(dst, Message1(val: 123))
  send(dst, Message1(val: 123))
  send(dst, Message4())


proc main() {.actor.} =

  discard hatch sender(self())


  block:
    receive:

      Message1(val: 123):
        echo "got Message1, val was a direct hit 123"

      (v, w) = Message1(val: v, weight: w):
        echo "got Message1, val was ", v, " weight ", w

      name = Message2(name: name):
        echo "got Message2, name was ", name

      Message3():
        echo "Got Message3"
      
      Message4():
        echo "Got Message4, bye"
        break


  echo "all good"


var pool = newPool(4)
discard pool.hatch main()
pool.join()
