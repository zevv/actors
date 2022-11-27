
import os
import strformat
import std/macros
import std/locks
import std/rlocks
import std/deques
import std/tables
import std/posix
import std/atomics
import std/times

import cps

import bitline
import actorid
import mailbox
import isisolated
import mallinfo


# FFI for glib mallinfo()

type 

  Pool* = object

    # Used to assign unique ActorIds
    actorIdCounter: Atomic[int]

    # All workers in the pool. No lock needed, only main thread touches this
    workers: seq[ref Worker]

    # This is where the continuations wait when not running
    workLock: Lock
    stop: bool
    workCond: Cond
    workQueue: Deque[Actor] # actor that needs to be run asap on any worker
    idleQueue: Table[ActorId, Actor] # actor that is waiting for messages
    killReq: Table[ActorId, bool]

    # mailboxes for the actors
    mailhub*: MailHub

  Actor* = ref object of Continuation
    id*: ActorId
    parentId*: ActorId
    pool*: ptr Pool
    links: seq[ActorId]

  Worker = object
    id: int
    thread: Thread[ptr Worker]
    pool: ptr Pool

  ExitReason* = enum
    erNormal, erKilled, erError

  MessageKill* = ref object of Message
  
  MessageExit* = ref object of Message
    id*: ActorId
    reason*: ExitReason
    ex*: ref Exception


# Forward declerations

proc kill*(pool: ptr Pool, id: ActorId)


# Misc helpers

proc `$`*(pool: ref Pool): string =
  return "#POOL<>"

proc `$`*(a: Actor): string =
  return "#ACT<" & $(a.parent_id.int) & "." & $(a.id.int) & ">"

proc `$`*(worker: ref Worker | ptr Worker): string =
  return "#WORKER<" & $worker.id & ">"


# Misc helper procs

proc pass*(cFrom, cTo: Actor): Actor =
  cTo.pool = cFrom.pool
  cTo.id = cFrom.id
  cTo


# Send a message from srcId to dstId

proc send*(pool: ptr Pool, srcId, dstId: ActorId, msg: sink Message) =

  assertIsolated(msg)

  pool.mailhub.sendTo(srcId, dstId, msg)

  # If the target continuation is in the sleep queue, move it to the work queue
  withLock pool.workLock:
    if dstId in pool.idleQueue:
      #echo "wake ", dstId
      let actor = pool.idleQueue[dstId]
      pool.idleQueue.del(dstId)
      pool.workQueue.addLast(actor)
      pool.workCond.signal()


# Signal termination of an actor; inform the parent and kill any linked
# actors.

proc exit(actor: sink Actor, reason: ExitReason, ex: ref Exception = nil) =
  #assertIsolated(actor)  # TODO: cps refs child

  echo &"{actor.id} exit, reason: {reason}"
  if not ex.isNil:
    echo "Exception: ", ex.msg
    echo ex.getStackTrace()

  let pool = actor.pool
  pool.mailhub.unregister(actor.id)
  let msg = MessageExit(id: actor.id, reason: reason, ex: ex)
  pool.send(0.ActorId, actor.parent_id, msg)
  for id in actor.links:
    {.cast(gcsafe).}:
      pool.kill(id)


# Kill an actor

proc kill*(pool: ptr Pool, id: ActorId) =
  withLock pool.workLock:
    # Mark the actor as to-be-killed so it will be caught before trampolining
    # or when jielding
    pool.killReq[id] = true
    # Send the actor a message so it will wake up if it is in the idle pool
  pool.send(0.ActorId, id, MessageKill())


# Move actor to the idle queue

proc toIdleQueue*(pool: ptr Pool, actor: sink Actor) =
  #assertIsolated(actor) # TODO
  var killed = false
  withLock pool.workLock:
    if actor.id in pool.killReq:
      pool.killReq.del(actor.id)
      killed = true
    else:
      pool.idleQueue[actor.id] = actor

  if killed:
    exit(actor, erKilled)


proc waitForWork(pool: ptr Pool): Actor =
  while true:
    withLock pool.workLock:

      while pool.workQueue.len == 0 and not pool.stop:
        pool.workCond.wait(pool.workLock)

      if pool.stop:
        return nil
      else:
        return pool.workQueue.popFirst()


proc workerThread(worker: ptr Worker) {.thread.} =

  let pool = worker.pool
  let wid = "worker." & $worker.id

  while true:

    # Wait for actor or stop request

    bitline.logStart(wid & ".wait")
    var actor = pool.waitForWork()
    bitline.logStop(wid & ".wait")
    
    if actor.isNil:
      break
    
    #assertIsolated(actor)  # TODO: cps refs child

    # Trampoline the continuation

    bitline.log "worker." & $worker.id & ".run":
      try:
        {.cast(gcsafe).}: # Error: 'workerThread' is not GC-safe as it performs an indirect call here
          while not actor.isNil and not actor.fn.isNil:
            actor = actor.fn(actor).Actor
      except:
        actor.exit(erError, getCurrentException())

    # Cleanup if continuation has finished

    if actor.finished:
      actor.exit(erNormal)
      

proc hatchAux*(pool: ref Pool | ptr Pool, actor: sink Actor, parentId=0.ActorId, link=false): ActorId =

  assert not isNil(actor)
  assertIsolated(actor)

  pool.actorIdCounter += 1
  let id = pool.actorIdCounter.load().ActorID

  # Initialize actor
  actor.pool = pool[].addr
  actor.id = id
  actor.parentId = parentId

  if link:
    actor.links.add parentId

  # Register a mailbox for the actor
  pool.mailhub.register(actor.id)

  # Add the new actor to the work queue
  withLock pool.workLock:
    assertIsolated(actor)
    pool.workQueue.addLast actor
    pool.workCond.signal()

  id


# Create and initialize a new actor

template hatch*(pool: ref Pool, c: typed): ActorId =
  var actor = Actor(whelp c)
  hatchAux(pool, actor)


proc setMailboxFd*(pool: ref Pool | ptr Pool, id: ActorId, fd: cint) {.cpsVoodoo.} =
  pool.mailhub.setSignalFd(id, fd)


# Create pool with actor queue and worker threads

proc newPool*(nWorkers: int): ref Pool =

  var pool = new Pool
  initLock pool.workLock
  initCond pool.workCond

  for i in 0..<nWorkers:
    var worker = new Worker
    worker.id = i
    worker.pool = pool[].addr
    pool.workers.add worker
    createThread(worker.thread, workerThread, worker[].addr)

  pool


# Wait until all actors in the pool have died and cleanup

proc join*(pool: ref Pool) =

  while pool.mailhub.len > 0:
    let mi = mallinfo2()
    bitline.logValue("stats.mailboxes", pool.mailhub.len)
    bitline.logValue("stats.mem_alloc", mi.uordblks)
    bitline.logValue("stats.mem_arena", mi.arena)
    os.sleep(10)

  echo "all mailboxes gone"

  withLock pool.workLock:
    pool.stop = true
    pool.workCond.broadcast()

  echo "waiting for workers to join"

  for worker in pool.workers:
    worker.thread.joinThread()
    assertIsolated(worker)

  echo "all workers stopped"


