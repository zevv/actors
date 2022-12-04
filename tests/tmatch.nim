

import std/macros
import nimactors

type
  Message1 = ref object of Message
    val: int

  Message2 = ref object of Message
    name: string

  Message3 = ref object of Message
    thing: float


proc main() {.actor.} =

  # Send some stuff; note this goes in the wrong order
 
  send(self(), Message3(thing: 3.14))
  send(self(), Message2(name: "charlie"))
  send(self(), Message1(val: 124))
  send(self(), Message1(val: 123))

  # See what we can match
 
  var val = 124

  while true:

    receive:

      Message1(val: 123):
        echo "got Message1, val was a direct hit 123"
      
      Message1(val: val):
        echo "got Message1, val was ", val
      
      Message2(name: "john"):
        echo "got Message2, val was "

      Message3():
        echo "Got Message3"
        echo msg.thing

  echo "all good"


var pool = newPool()
discard pool.hatch main()
pool.join()
