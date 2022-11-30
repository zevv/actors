
import std/locks
import std/macros
import std/deques
import std/atomics

import cps

import pool
import actorobj
import isisolated
 
macro actor*(n: untyped): untyped =
  n.addPragma nnkExprColonExpr.newTree(ident"cps", ident"ActorCont")
  n     


# Yield the continuation by storing it back into the actor object; it can later
# be resumed by calling toWorkQueue

proc suspend*(c: sink ActorCont): ActorCont {.cpsMagic.} =
  c.actor.suspend(c)


# Move a running process to the back of the work queue

proc jield*(c: ActorCont): ActorCont {.cpsMagic.} =
  if not c.actor[].killReq.load():
    result = c


# Receive a message, nonblocking

proc tryRecv*(c: ActorCont): Message {.cpsVoodoo.} =
  result = c.actor.tryRecv2()

proc tryRecv*(c: ActorCont, srcId: Actor): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg.src == srcId
  result = c.actor.tryRecv2(filter)

proc tryRecv*(c: ActorCont, T: typedesc): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg of T
  result = c.actor.tryRecv2(filter)

# Receive a message, blocking

template recv*(): Message =
  var msg: Message = nil
  while msg.isNil:
    msg = tryRecv()
    if msg.isNil:
      suspend()
  msg

template recv*(T: typedesc): auto =
  var msg: Message = nil
  while msg.isNil:
    msg = tryRecv(T)
    if msg.isNil:
      suspend()
  T(msg)


# Send a message to another actor

proc send*(c: ActorCont, dst: Actor, msg: sink Message) {.cpsVoodoo.} =
  c.pool.send(c.actor, dst, msg)


# Hatch an actor from within an actor

proc hatchAux*(c: ActorCont, newActor: sink ActorCont, link: bool): Actor {.cpsVoodoo.} =
  assertIsolated(c, 1)
  c.pool.hatchAux(newActor, c.actor, link)


# Hatches the given actor and returns its AID.

template hatch*(what: typed): Actor =
  hatchAux(ActorCont(whelp what), false)


# Hatches the given actor passing, links it to the current process, and returns
# its PID.

template hatchLinked*(what: typed): Actor =
  hatchAux(ActorCont(whelp what))


# Returns the AID of the calling actor

proc self*(c: ActorCont): Actor {.cpsVoodoo.} =
  c.actor


# Kill

proc kill*(c: ActorCont, actor: Actor) {.cpsVoodoo.} =
  c.pool.kill(actor)


