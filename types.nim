
import os
import strformat
import std/macros
import std/locks
import std/deques
import std/tables
import std/posix
import std/atomics
import std/times

import cps

import bitline
import isisolated

type

  ActorId* = int

  Actor* = ref object of Continuation
    id*: ActorId
    parentId*: ActorId
    pool*: ptr Pool

  Worker* = object
    id*: int
    thread*: Thread[ptr Worker]
    pool*: ptr Pool

  Mailbox*[T] = ref object
    lock*: Lock
    queue*: Deque[T]

  MailHub* = object
    lock*: Lock
    table*: Table[ActorId, Mailbox[Message]]

  Pool* = object

    # Used to assign unique ActorIds
    actorIdCounter*: Atomic[int]

    # All workers in the pool. No lock needed, only main thread touches this
    workers*: seq[ref Worker]

    # This is where the continuations wait when not running
    workLock*: Lock
    stop*: bool
    workCond*: Cond
    workQueue*: Deque[Actor] # actor that needs to be run asap on any worker
    idleQueue*: Table[ActorId, Actor] # actor that is waiting for messages

    # mailboxes for the actors
    mailhub*: MailHub

    # Event queue glue. please ignore
    evqActorId*: ActorId
    evqFdWake*: cint

  Message* = ref object of Rootobj
    src*: ActorId

  MessageDied* = ref object of Message
    id*: ActorId


# Misc helper procs

proc `$`*(pool: ref Pool): string =
  return "#POOL<>"

proc `$`*(worker: ref Worker | ptr Worker): string =
  return "#WORKER<" & $worker.id & ">"

proc `$`*(a: Actor): string =
  return "#ACT<" & $a.parent_id & "." & $a.id & ">"

proc `$`*(m: Message): string =
  return "#MSG<" & $m.src & ">"

