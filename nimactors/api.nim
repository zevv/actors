
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


# Move the given actor to the idle queue. It will only be moved aback to the
# workQueue whenever a new message arrives

proc toIdleQueue*(actor: sink Actor): Actor {.cpsMagic.} =
  actor.pool.toIdleQueue(actor)


# Receive a message, nonblocking

proc tryRecv*(actor: Actor): Message {.cpsVoodoo.} =
  result = actor.pool.mailhub.tryRecv(actor.id)
  if result of MessageKill:
    result = nil # will cause a jield, catching the kill

proc tryRecv*(actor: Actor, srcId: ActorId): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg.src == srcId
  result = actor.pool.mailhub.tryRecv(actor.id, filter)
  if result of MessageKill:
    result = nil # will cause a jield, catching the kill

proc tryRecv*(actor: Actor, T: typedesc): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg of T
  result = actor.pool.mailhub.tryRecv(actor.id, filter)
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

proc send*(actor: Actor, dst: ActorId, msg: sink Message) {.cpsVoodoo.} =
  actor.pool.send(actor.id, dst, msg)


# Send a kill message to another actor

proc kill*(actor: Actor, dst: ActorId) {.cpsVoodoo.} =
  actor.pool.kill(dst)


# Hatch an actor from within an actor

proc hatchAux*(actor: Actor, newActor: sink Actor, link: bool): ActorId {.cpsVoodoo.} =
  actor.pool.hatchAux(newActor, actor.id, link)


# Hatches the given actor and returns its AID.

template hatch*(c: typed): ActorId =
  let actor = Actor(whelp c)
  hatchAux(actor, false)


# Hatches the given actor passing, links it to the current process, and returns
# its PID.

template hatchLinked*(c: typed): ActorId =
  let actor = Actor(whelp c)
  hatchAux(actor, true)


# Returns the AID of the calling actor

proc self*(actor: Actor): ActorId {.cpsVoodoo.} =
  actor.id

# Register a signaling file descriptor for this actors mailbox

proc setMailboxFd*(actor: Actor, fd: cint) {.cpsVoodoo.} =
  actor.pool.mailhub.setSignalFd(actor.id, fd)

proc setMailboxFd*(actor: Actor, id: ActorId, fd: cint) {.cpsVoodoo.} =
  actor.pool.mailhub.setSignalFd(id, fd)

