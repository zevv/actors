
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

  MailFilter* = proc(msg: Message): bool

  Message* = ref object of Rootobj
    src*: ActorId

proc `$`*(m: Message): string =
  return "#MSG<" & $m.src & ">"


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


# Try to receive one message from the mailbox. If a filter proc is given,
# get the first message matching the filter.

proc tryRecv*(mailhub: var Mailhub, id: ActorId, filter: MailFilter = nil): Message =
  withMailbox(mailhub, id):
    var first = true
    for msg {.inject.} in mailbox.queue.mitems:
      if not msg.isNil and (filter.isNil or filter(msg)):
        result = msg
        if first:
          mailbox.queue.popFirst()
        else:
          msg = nil
        break
      first = false

