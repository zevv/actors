

import std/atomics
import std/locks
import std/posix
import std/strformat
import std/deques

import isisolated
import bitline

import cps

type

  State* = enum
    Idle, Running, Dead

  ActorObject* = object
    rc*: Atomic[int]
    state*: Atomic[State]
    killReq*: Atomic[bool]
    c*: Continuation
    pid*: int
    parent*: Actor

    lock*: Lock
    links*: seq[Actor]
    mailBox*: Deque[Message]
    signalFd*: cint

  Actor* = object
    p*: ptr ActorObject

  Message* = ref object of Rootobj
    src*: Actor

  MessageKill* = ref object of Message
  
  MessageExit* = ref object of Message
    id*: Actor
    reason*: ExitReason
    ex*: ref Exception

  ExitReason* = enum
    erNormal, erKilled, erError
  
  MailFilter* = proc(msg: Message): bool



template log(a: Actor, msg: string) =
  #echo "\e[1;35m" & $a & ": " & msg & "\e[0m"
  discard

proc `=destroy`*(actor: var Actor)

proc `=copy`*(dest: var Actor, actor: Actor) =
  if not actor.p.isNil:
    actor.p[].rc.atomicInc()
    actor.log("rc ++ " & $actor.p[].rc.load())
  if not dest.p.isNil:
    `=destroy`(dest)
  dest.p = actor.p


proc `=destroy`*(actor: var Actor) =
  if not actor.p.isNil:
    if actor.p[].rc.load(moAcquire) == 0:
      actor.log("destroy")
      `=destroy`(actor.p[])
      deallocShared(actor.p)
    else:
      actor.p[].rc.atomicDec()
      actor.log("rc -- " & $actor.p[].rc.load())


proc `[]`*(actor: Actor): var ActorObject =
  assert not actor.p.isNil
  actor.p[]


proc isNil*(actor: Actor): bool =
  actor.p.isNil


# Move the continuation back into the actor; the actor can later be
# resumed to continue the continuation

proc suspend*(actor: Actor, c: sink Continuation) =
  var exp = Running
  if actor[].state.compareExchange(exp, Idle):
    actor[].c = move c
  else:
    echo "not moving to idle, state was ", exp


proc newActor*(pid: int, parent: Actor): Actor =
  let a = create(ActorObject)
  a.pid = pid
  a.rc.store(0)
  a.lock.initLock()
  a.parent = parent
  a.state.store(Idle)
  result.p = a
  result.log("new")


template withLock*(actor: Actor, code: untyped) =
  if not actor.p.isNil:
    withLock actor[].lock:
      code
  else:
    echo "empty actor"


proc `$`*(a: Actor): string =
  if not a.p.isNil:
    "actor." & $a.p[].pid
  else:
    "actor.nil"


proc `$`*(m: Message): string =
  if not m.isNil:
    return "msg src=" & $m.src
  else:
    return "msg.nil"


proc link*(a, b: Actor) =
  withLock a:
    a[].links.add b
  withLock b:
    b[].links.add a


# Receive a message

proc tryRecv2*(actor: Actor, filter: MailFilter = nil): Message =
  withLock actor:
    var first = true
    for msg in actor[].mailbox.mitems:
      if not msg.isNil and (filter.isNil or filter(msg)):
        result = msg
        if first:
          actor[].mailbox.popFirst()
        else:
          msg = nil
        break
      first = false
  #echo &"  tryRecv {actor}: {result}"


proc sendAux*(dst: Actor, src: Actor, msg: sink Message) =
  #assertIsolated(msg)
  #echo &"  send {src} -> {dst}: {msg}"
  msg.src = src

  # Deliver the message in the target mailbox
  var count: int
  withLock dst:
    dst[].mailbox.addLast(msg)
    count = dst[].mailbox.len
  bitline.logValue("actor." & $dst & ".mailbox", count)


proc setSignalFd*(actor: Actor, fd: cint) =
  withLock actor:
    actor[].signalFd = fd


