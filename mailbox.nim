
import os
import strformat
import std/macros
import std/locks
import std/deques
import std/tables
import std/posix
import std/atomics
import std/times

import actorid
import isisolated
import bitline

type
  
  Mailbox*[T] = ref object
    lock*: Lock
    queue*: Deque[T]

  MailHub* = object
    lock*: Lock
    table*: Table[ActorId, Mailbox[Message]]

  Message* = ref object of Rootobj
    src*: ActorId

  MessageDied* = ref object of Message
    id*: ActorId

proc `$`*(m: Message): string =
  if not m.isNil:
    "#MSG<" & $(m.src.int) & ">"
  else:
    "nil"


# Get number of mailboxes in a mailhub

proc len*(mailhub: var Mailhub): int =
  withLock mailhub.lock:
    result = mailhub.table.len


# Create a new mailbox with the given id

proc register*(mailhub: var Mailhub, id: ActorId) =
  var mailbox = new Mailbox[Message]
  initLock mailbox.lock
  withLock mailhub.lock:
    mailhub.table[id] = mailbox


# Unregister / destroy a mailbox from the hub

proc unregister*(mailhub: var Mailhub, id: ActorId) =
  withLock mailhub.lock:
    mailhub.table.del(id)


# Do something with the given mailbox while holding the proper locks

template withMailbox(mailhub: var Mailhub, id: ActorId, code: untyped) =
  withLock mailhub.lock:
    if id in mailhub.table:
      var mailbox {.cursor,inject.} = mailhub.table[id]
      withLock mailbox.lock:
        code


# Deliver message in the given mailbox

proc sendTo*(mailhub: var Mailhub, srcId, dstId: ActorID, msg: sink Message) =
  msg.src = srcId
  #echo &"  send {srcId} -> {dstId}: {msg.repr}"
  mailhub.withMailbox(dstId):
    assertIsolated(msg)
    mailbox.queue.addLast(msg)
    bitline.logValue("actor." & $dstId & ".mailbox", mailbox.queue.len)


# Check for mail, returns nil if nothing found. If the first item in the
# dequeue matches the filter, pop it. If it is not the first item, nil it
# and leave a hole.

template tryRecvFilterIt*(mailhub: var Mailhub, id: ActorId, filter: untyped): Message =
  var msg: Message
  withMailbox(mailhub, id):
    var i = 0
    for it {.inject.} in mailbox.queue.mitems:
      if not it.isNil and filter:
        msg = it
        if i == 0:
          mailbox.queue.popFirst()
        else:
          it = nil
        break
      inc i
  msg

proc tryRecv*(mailhub: var Mailhub, id: ActorId): Message =
  tryRecvFilterIt mailhub, id:
    true

proc tryRecv*(mailhub: var Mailhub, id: ActorId, idSrc: ActorId): Message =
  tryRecvFilterIt mailhub, id:
    it.src == idSrc

proc tryRecv*(mailhub: var Mailhub, id: ActorId, T: typedesc): Message =
  tryRecvFilterIt mailhub, id:
    it of T

proc tryRecv*(mailhub: var Mailhub, id: ActorId, idSrc: ActorId, T: typedesc): Message =
  tryRecvFilterIt mailhub, id:
    it.src == idSrc and it of T


proc dump*(mailhub: var Mailhub, id: ActorId) =
  mailhub.withMailbox(id):
    var i = 0
    for m in mailbox.queue.items:
      echo i, ": ", m
      inc i

