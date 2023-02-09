
import std/locks
import std/macros
import std/os
import std/strutils
import std/deques
import std/atomics

import cps

import pool
import isisolated
 

# CPS `{.actor.}` macro

macro actor*(n: untyped): untyped =
  n.addPragma nnkExprColonExpr.newTree(ident"cps", ident"ActorCont")
  n     


# Hatch an actor from within an actor

proc hatchAux*(c: ActorCont, newActor: sink ActorCont, link: bool): Actor {.cpsVoodoo.} =
  c.pool.hatchAux(newActor, c.actor, link)


# Create and initialize a new actor

template hatch*(pool: ref Pool, what: typed): Actor =
  hatchAux(pool[].addr, ActorCont(whelp what))


# Hatches the given actor and returns its AID.

template hatch*(what: typed): Actor =
  hatchAux(ActorCont(whelp what), false)


# Hatches the given actor passing, links it to the current process, and returns
# its PID.

template hatchLinked*(what: typed): Actor =
  hatchAux(ActorCont(whelp what), true)


# Returns the `self` reference of the calling actor

proc self*(c: ActorCont): Actor {.cpsVoodoo.} =
  c.actor


# Starts monitoring the given actor

proc monitor*(c: ActorCont, slave: Actor) {.cpsVoodoo.} =
  c.actor.monitor(slave)


# Yield the continuation by storing it back into the actor object; it can later
# be resumed by calling toWorkQueue

proc suspend*(c: sink ActorCont): ActorCont {.cpsMagic.} =
  if c.actor.trySuspend(c):
    return nil
  else:
    return c


# Move a running process to the back of the work queue

proc jield*(c: sink ActorCont): ActorCont {.cpsMagic.} =
  c.actor.jield(c)


# Send a message to another actor

proc sendAux*(c: ActorCont, dst: Actor, msg: sink Message) {.cpsVoodoo.} =
  dst.sendSig(msg, c.actor)

template send*(dst: Actor, msg: var typed) =
  assertIsolated(msg)
  dst.sendAux(move msg)

template send*(dst: Actor, msg: typed) =
  assertIsolated(msg)
  var msgCopy = msg
  dst.sendAux(move msgCopy)

# Get message number `idx` from the actors message queue, returns nil
# if no such message
#
proc getMsg*(c: ActorCont, idx: Natural): Message {.cpsVoodoo.} =
  let actor = c.actor
  if idx < actor[].msgQueue.len:
    result = actor[].msgQueue[idx]


proc dropMsg*(c: ActorCont) {.cpsVoodoo.} =
  c.actor[].msgQueue.shrink(1)


# Try to receive a message, returns `nil` if no messages available or matched
# the passed filter

proc tryRecv*(actor: Actor, filter: MailFilter = nil): Message =
  var first = true
  for msg in actor[].msgQueue.mitems:
    if not msg.isNil and (filter.isNil or filter(msg)):
      result = msg
      if first:
        actor[].msgQueue.popFirst()
      else:
        msg = nil
      break
    first = false

# Receive a message, nonblocking

proc tryRecv*(c: ActorCont): Message {.cpsVoodoo.} =
  c.actor.tryRecv()

proc tryRecv*(c: ActorCont, srcId: Actor): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg.src == srcId
  c.actor.tryRecv(filter)

proc tryRecv*(c: ActorCont, T: typedesc): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg of T
  c.actor.tryRecv(filter)

proc tryRecv*(c: ActorCont, filter: MailFilter): Message {.cpsVoodoo.} =
  c.actor.tryRecv(filter)


# Receive a message, blocking

template recv*(): Message =
  var msg: Message
  while true:
    msg = tryRecv()
    if msg.isNil:
      suspend()
    else:
      break
  move msg

template recv*(filter: MailFilter): Message =
  var msg: Message
  while true:
    msg = tryRecv(filter)
    if msg.isNil:
      suspend()
    else:
      break
  move msg

template recv*(T: typedesc): auto =
  var msg: Message
  while true:
    msg = tryRecv(T)
    if msg.isNil:
      suspend()
    else:
      break
  T(move msg)
