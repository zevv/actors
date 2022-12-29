
import std/os
import std/sugar

import actors

type
  MessageOne = ref object of Message

  MessageTwo = ref object of Message
    val: int


proc test() {.actor.} =

  let me = self()

  block:

    me.send(MessageOne())
    let m = recv()
    doAssert m of MessageOne
  
  block:

    me.send(MessageOne())
    me.send(MessageTwo())
    let m = recv(MessageTwo)
    doAssert m of MessageTwo
    doAssert recv() of MessageOne
 
  block:

    me.send(MessageOne())
    me.send(MessageTwo())

    proc filter(m: Message): bool = m of MessageTwo
    doAssert recv(filter) of MessageTwo
    doAssert recv() of MessageOne
  
  when false:
    # fails to compile in CPS
    me.send(MessageOne())
    me.send(MessageTwo())
    doAssert recv((m) => m of MessageTwo) of MessageTwo
    doAssert recv() of MessageOne
  



  echo "all good"


proc main() {.actor.} =
  let f = hatch test()
  let m = recv().MessageExit
  echo m.reason
  if m.reason == Error:
    echo m.ex.msg
  


var pool = newPool(4)
discard pool.hatch main()
pool.join()

