
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


proc jield*(actor: sink Actor): Actor {.cpsMagic.} =
  actor.pool.jieldActor(actor)


proc tryRecv*(actor: Actor): Message {.cpsVoodoo.} =
  result = actor.pool.mailhub.tryRecv(actor.id)


template recv*(): Message =
  # TODO: why the need to set to nil?
  var msg: Message = nil
  while msg.isNil:
    msg = tryRecv()
    if msg.isNil:
      jield()
  msg


proc send*(actor: Actor, dst: ActorId, msg: sink Message) {.cpsVoodoo.} =
  actor.pool.send(actor.id, dst, msg)


# Hatch an actor from within an actor

proc hatchFromActor*(actor: Actor, newActor: sink Actor): ActorId {.cpsVoodoo.} =
  hatchAux(actor.pool, newActor, actor.id)


# Create and initialize a new actor from within an actor

template hatch*(c: typed): ActorId =
  let actor = Actor(whelp c)
  hatchFromActor(actor)

