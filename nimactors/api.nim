
import std/locks
import std/macros

import cps

import actors
import isisolated
 
macro actor*(n: untyped): untyped =
  n.addPragma nnkExprColonExpr.newTree(ident"cps", ident"ActorCont")
  n     


# Move the given actor to the idle queue. It will only be moved aback to the
# workQueue whenever a new message arrives

proc toIdleQueue*(c: sink ActorCont): ActorCont {.cpsMagic.} =
  c.pool.toIdleQueue(c)


# Receive a message, nonblocking

proc tryRecv*(c: ActorCont): Message {.cpsVoodoo.} =
  result = c.pool.tryRecv(c.actor)
  if result of MessageKill:
    result = nil # will cause a jield, catching the kill

proc tryRecv*(c: ActorCont, srcId: Actor): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg.src == srcId
  result = c.pool.tryRecv(c.actor, filter)
  if result of MessageKill:
    result = nil # will cause a jield, catching the kill

proc tryRecv*(c: ActorCont, T: typedesc): Message {.cpsVoodoo.} =
  proc filter(msg: Message): bool = msg of T
  result = c.pool.tryRecv(c.actor, filter)
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

proc send*(c: ActorCont, dst: Actor, msg: sink Message) {.cpsVoodoo.} =
  c.pool.send(c.actor, dst, msg)


# Send a kill message to another actor

proc kill*(c: ActorCont, dst: Actor) {.cpsVoodoo.} =
  c.pool.kill(dst)


# Hatch an actor from within an actor

proc hatchAux*(c: ActorCont, newActor: sink ActorCont, link: bool): Actor {.cpsVoodoo.} =
  c.pool.hatchAux(newActor, c.actor, link)


# Hatches the given actor and returns its AID.

template hatch*(what: typed): Actor =
  let c = ActorCont(whelp what)
  hatchAux(c, false)


# Hatches the given actor passing, links it to the current process, and returns
# its PID.

template hatchLinked*(what: typed): Actor =
  let c = ActorCont(whelp what)
  hatchAux(c, true)


# Returns the AID of the calling actor

proc self*(c: ActorCont): Actor {.cpsVoodoo.} =
  c.actor

# Register a signaling file descriptor for this actors mailbox

proc setMailboxFd*(c: ActorCont, fd: cint) {.cpsVoodoo.} =
  c.pool.setSignalFd(c.actor, fd)

proc setMailboxFd*(c: ActorCont, id: Actor, fd: cint) {.cpsVoodoo.} =
  c.pool.setSignalFd(id, fd)

