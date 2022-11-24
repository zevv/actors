
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

