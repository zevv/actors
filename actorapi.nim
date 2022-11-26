
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


# Receive a message, non-blocking

proc tryRecv*(actor: Actor, filter: MailFilter): Message {.cpsVoodoo.} =
  result = actor.pool.mailhub.tryRecv(actor.id, filter)


# Receive a message, yield if necessary

when false:

  # This should work but breaks assertIsolated() for the actors
  proc recv*(filter: MailFilter = nil): Message {.actor.} =
    while result.isNil:
      result = tryRecv(filter)
      if result.isNil:
        jield()
  
  proc recv*(T: typedesc): Message {.actor.} =
    result = recv()

else:

  template recv*(filter: MailFilter = nil): Message =
    var msg: Message = nil
    while msg.isNil:
      msg = tryRecv(filter)
      if msg.isNil:
        jield()
    msg

  #template recv*(id: ActorId): Message =
  #  proc filter(msg: Message): bool =
  #    echo "filter"
  #  var msg: Message = nil
  #  while msg.isNil:
  #    msg = tryRecv(filter)
  #    if msg.isNil:
  #      jield()
  #  msg


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


