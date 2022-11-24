
import os
import strformat
import std/macros
import std/locks
import std/deques
import std/tables
import std/posix
import std/atomics
import std/times

import types
import isisolated
import bitline

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

template withMailbox*(mailhub: var Mailhub, id: ActorId, code: untyped) =
  withLock mailhub.lock:
    if id in mailhub.table:
      var mailbox {.cursor,inject.} = mailhub.table[id]
      withLock mailbox.lock:
        code

proc sendTo*(mailhub: var Mailhub, srcId, dstId: ActorID, msg: sink Message) =
  msg.src = srcId
  echo &"  send {srcId} -> {dstId}: {msg.repr}"
  mailhub.withMailbox(dstId):
    assertIsolated(msg)
    mailbox.queue.addLast(msg)
    bitline.logValue("actor." & $dstId & ".mailbox", mailbox.queue.len)

  

proc tryRecv*(mailhub: var Mailhub, id: ActorId): Message =
  var len: int
  mailhub.withMailbox(id):
    len = mailbox.queue.len
    if len > 0:
      result = mailbox.queue.popFirst()
      assertIsolated(result)
      bitline.logValue("actor." & $id & ".mailbox", len)

