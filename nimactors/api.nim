
import std/locks
import std/macros

import cps

import actorid
import actors
import isisolated
 
macro actor*(n: untyped): untyped =
  n.addPragma nnkExprColonExpr.newTree(ident"cps", ident"ActorCond")
  n     


# Move the given actor to the idle queue. It will only be moved aback to the
# workQueue whenever a new message arrives

proc toIdleQueue*(actor: sink ActorCond): ActorCond {.cpsMagic.} =
  actor.pool.toIdleQueue(actor)


# Receive a message, nonblocking

proc tryRecv*(actor: ActorCond): Message {.cpsVoodoo.} =
  result = actor.pool.tryRecv(actor.id)
  if result of MessageKill:
    result = nil # will cause a jield, catching the kill

proc tryRecv*(actor: ActorCond, srcId: Actor): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg.src == srcId
  result = actor.pool.tryRecv(actor.id, filter)
  if result of MessageKill:
    result = nil # will cause a jield, catching the kill

proc tryRecv*(actor: ActorCond, T: typedesc): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg of T
  result = actor.pool.tryRecv(actor.id, filter)
  if result of MessageKill:
    result = nil # will cause a jield, catching the kill

# Receive a message, blocking

template recv*(): Message =
  var msg: Message = nil
  while msg.isNil:
    msg = tryRecv()
    if msg.isNil:
      toIdleQueue()
  msg

template recv*(T: typedesc): auto =
  var msg: Message = nil
  while msg.isNil:
    msg = tryRecv(T)
    if msg.isNil:
      toIdleQueue()
  T(msg)


# Send a message to another actor

proc send*(actor: ActorCond, dst: Actor, msg: sink Message) {.cpsVoodoo.} =
  actor.pool.send(actor.id, dst, msg)


# Send a kill message to another actor

proc kill*(actor: ActorCond, dst: Actor) {.cpsVoodoo.} =
  actor.pool.kill(dst)


# Hatch an actor from within an actor

proc hatchAux*(actor: ActorCond, newActor: sink ActorCond, link: bool): Actor {.cpsVoodoo.} =
  actor.pool.hatchAux(newActor, actor.id, link)


# Hatches the given actor and returns its AID.

template hatch*(c: typed): Actor =
  let actor = ActorCond(whelp c)
  hatchAux(actor, false)


# Hatches the given actor passing, links it to the current process, and returns
# its PID.

template hatchLinked*(c: typed): Actor =
  let actor = ActorCond(whelp c)
  hatchAux(actor, true)


# Returns the AID of the calling actor

proc self*(actor: ActorCond): Actor {.cpsVoodoo.} =
  actor.id

# Register a signaling file descriptor for this actors mailbox

proc setMailboxFd*(actor: ActorCond, fd: cint) {.cpsVoodoo.} =
  actor.pool.setSignalFd(actor.id, fd)

proc setMailboxFd*(actor: ActorCond, id: Actor, fd: cint) {.cpsVoodoo.} =
  actor.pool.setSignalFd(id, fd)

