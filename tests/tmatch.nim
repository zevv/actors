

import std/macros
import nimactors

type
  Message1 = ref object of Message
    val: int
    weight: float

  Message2 = ref object of Message
    name: string

  Message3 = ref object of Message
    thing: float


proc main() {.actor.} =

  # Send some stuff; note this goes in the wrong order
 
  send(self(), Message1(val: 122, weight: 14.4))
  send(self(), Message3(thing: 3.14))
  send(self(), Message2(name: "charlie"))
  send(self(), Message1(val: 124))
  send(self(), Message1(val: 123))

  # See what we can match
 
  while true:

    receive:

      Message1(val: 123):
        echo "got Message1, val was a direct hit 123"
      
      (v, w) = Message1(val: v, weight: w):
        echo "got Message1, val was ", v, " weight ", w
      
      name = Message2(name: name):
        echo "got Message2, name was ", name

      Message3():
        echo "Got Message3"
        #echo msg.thing

      else:
        suspend()

  echo "all good"


var pool = newPool()
discard pool.hatch main()
pool.join()
