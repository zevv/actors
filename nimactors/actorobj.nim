

import std/atomics
import std/locks
import std/posix
import std/strformat
import std/deques

import isisolated
import bitline

type

  ActorObject* = object
    rc*: Atomic[int]
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


proc `=copy`*(dest: var Actor, ai: Actor) =
  if not ai.p.isNil:
    ai.p[].rc.atomicInc()
    ai.log("rc ++ " & $ai.p[].rc.load())
  doAssert dest.p.isNil
  dest.p = ai.p


proc `=destroy`*(ai: var Actor) =
  if not ai.p.isNil:
    if ai.p[].rc.load(moAcquire) == 0:
      ai.log("destroy")
      `=destroy`(ai.p[])
      deallocShared(ai.p)
    else:
      ai.p[].rc.atomicDec()
      ai.log("rc -- " & $ai.p[].rc.load())


proc `[]`*(ai: Actor): var ActorObject =
  assert not ai.p.isNil
  ai.p[]



proc newActor*(): Actor =
  result.p = create(ActorObject)
  #echo "ai: new ", cast[int](result.p)
  result.p[].rc.store(0)
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
    return "msg"
  else:
    return "nil"

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
  #echo &"  tryRecv {id}: {result}"


proc send*(dst: Actor, src: Actor, msg: sink Message) =
  assertIsolated(msg)
  #echo &"  send {src} -> {dst}: {msg}"
  msg.src = src

  # Deliver the message in the target mailbox
  withLock dst:
    dst[].mailbox.addLast(msg)
    bitline.logValue("actor." & $dst & ".mailbox", dst[].mailbox.len)
    # If the target has a signalFd, wake it
    if dst[].signalFd != 0.cint:
      let b: char = 'x'
      discard posix.write(dst[].signalFd, b.addr, 1)

