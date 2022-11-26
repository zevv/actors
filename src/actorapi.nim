
import std/locks
import std/macros

import cps

import actorid
import actors
import isisolated
import mailbox
 
macro actor*(n: untyped): untyped =
  n.addPragma nnkExprColonExpr.newTree(ident"cps", ident"Actor")
  n     


# Yield

proc jield*(actor: sink Actor): Actor {.cpsMagic.} =
  actor.pool.jieldActor(actor)


# Receive a message, nonblocking

proc tryRecv*(actor: Actor): Message {.cpsVoodoo.} =
  result = actor.pool.mailhub.tryRecv(actor.id)

proc tryRecv*(actor: Actor, srcId: ActorId): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg.src == srcId
  result = actor.pool.mailhub.tryRecv(actor.id, filter)

proc tryRecv*(actor: Actor, T: typedesc): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg of T
  result = actor.pool.mailhub.tryRecv(actor.id, filter)

# Receive a message, blocking

template recv*(): Message =
  var msg: Message = nil
  while msg.isNil:
    msg = tryRecv()
    if msg.isNil:
      jield()
  msg

template recv*(T: typedesc): auto =
  var msg: Message = nil
  while msg.isNil:
    msg = tryRecv(T)
    if msg.isNil:
      jield()
  T(msg)

# Send a message to another actor

proc send*(actor: Actor, dst: ActorId, msg: sink Message) {.cpsVoodoo.} =
  actor.pool.send(actor.id, dst, msg)


# Hatch an actor from within an actor

proc hatchFromActor*(actor: Actor, newActor: sink Actor): ActorId {.cpsVoodoo.} =
  hatchAux(actor.pool, newActor, actor.id)


# Create and initialize a new actor from within an actor

template hatch*(c: typed): ActorId =
  let actor = Actor(whelp c)
  hatchFromActor(actor)


# Returns the ActorID of the calling actor

proc self*(actor: Actor): ActorId {.cpsVoodoo.} =
  actor.id


