
import os
import strformat
import std/locks
import std/deques
import std/tables
import std/atomics
import std/times

import cps

import bitline
import isisolated


type

  ActorId* = int

  Actor* = ref object of Continuation
    id: ActorId
    parentId: ActorId
    pool: ptr Pool

  Worker = ref object
    id: int
    thread: Thread[Worker]
    pool: ptr Pool

  Mailbox[T] = ref object
    lock: Lock
    queue: Deque[T]

  Pool* = object

    # Used to assign unique ActorIds
    actorIdCounter: Atomic[int]

    # All workers in the pool. No lock needed, only main thread touches this
    workers: seq[Worker]

    # This is where the continuations wait when not running
    workLock: Lock
    stop: bool
    workCond: Cond
    workQueue: Deque[Actor] # actor that needs to be run asap on any worker
    idleQueue: Table[ActorId, Actor] # actor that is waiting for messages

    # mailboxes for the actors
    mailhubLock: Lock
    mailhubTable: Table[ActorId, Mailbox[Message]]

  Message* = ref object of Rootobj
    src*: ActorId

  MessageDied* = ref object of Message
    id*: ActorId




proc pass*(cFrom, cTo: Actor): Actor =
  cTo.pool = cFrom.pool
  cTo.id = cFrom.id
  cTo


proc send(pool: ptr Pool, srcId, dstId: ActorId, msg: sink Message) =

  msg.src = srcId
  #echo &"  send {srcId} -> {dstId}: {msg.repr}"

  withLock pool.mailhubLock:
    if dstId in pool.mailhubTable:
      let mailbox = pool.mailhubTable[dstId]
      withLock mailbox.lock:
        mailbox.queue.addLast(msg)
        # msg.wasMoved() # was it or was it not?
        bitline.logValue("actor." & $dstId & ".mailbox", mailbox.queue.len)
    else:
      discard
      #raise ValueError.newException &"send: no mailbox for {dstId} found"

  # If the target continuation is in the sleep queue, move it to the work queue
  withLock pool.workLock:
    if dstId in pool.idleQueue:
      #echo "wake ", dstId
      var actor = pool.idleQueue[dstId]
      pool.idleQueue.del(dstId)
      pool.workQueue.addLast(actor)
      pool.workCond.signal()
      actor.wasMoved


proc sendAux*(actor: Actor, dst: ActorId, msg: sink Message) {.cpsVoodoo.} =
  actor.pool.send(actor.id, dst, msg)


template send*(dst: ActorId, msg: Message): typed =
  verifyIsolated(msg)
  sendAux(dst, msg)


proc recvYield*(actor: sink Actor): Actor {.cpsMagic.} =
  # If there are no messages waiting in the mailbox, move the continuation to
  # the idle queue. Otherwise, return the current continuation so it can
  # receive and handle the mail without yielding
  let pool = actor.pool
  withLock pool.mailhubLock:
    assert actor.id in pool.mailhubTable
    let mailbox {.cursor.} = pool.mailhubTable[actor.id]
    if mailbox.queue.len == 0:
      withLock pool.workLock:
        pool.idleQueue[actor.id] = actor
      actor.wasMoved()
    else:
      result = actor


proc recvGetMessage*(actor: Actor): Message {.cpsVoodoo.} =
  let pool = actor.pool
  withLock pool.mailhubLock:
    if actor.id in pool.mailhubTable:
      let mailbox {.cursor.} = pool.mailhubTable[actor.id]
      withLock mailbox.lock:
        result = mailbox.queue.popFirst()
        bitline.logValue("actor." & $actor.id & ".mailbox", mailbox.queue.len)
    else:
      raise ValueError.newException &"recv: no mailbox for {actor.id} found"


template recv*(): Message =
  recvYield()
  recvGetMessage()



proc workerThread(worker: Worker) {.thread.} =
  let pool = worker.pool
  let wid = "worker." & $worker.id

  while true:

    # Wait for actor or stop request

    bitline.logStart(wid & ".wait")

    var actor: Actor
    withLock pool.workLock:
      while pool.workQueue.len == 0 and not pool.stop:
        pool.workCond.wait(pool.workLock)
      if pool.stop:
        break
      actor = pool.workQueue.popFirst()

    bitline.logStop(wid & ".wait")

    # Trampoline once and push result back on the queue

    if not actor.fn.isNil:
      #echo "\e[35mtramp ", actor.id, " on worker ", worker.id, "\e[0m"
      {.cast(gcsafe).}: # Error: 'workerThread' is not GC-safe as it performs an indirect call here
        let aid = "actor." & $actor.id & ".run"
        let wid = "worker." & $worker.id & ".run"

        bitline.logStart(wid)
        bitline.logStart(aid)

        actor = trampoline(actor)

        bitline.logStop(wid)
        bitline.logStop(aid)

      if not isNil(actor) and isNil(actor.fn):
        echo &"actor {actor.id} has died, parent was {actor.parent_id}"
        # Delete the mailbox for this actor
        withLock pool.mailhubLock:
          pool.mailhubTable.del(actor.id)
        # Send a message to the parent
        if actor.parent_id > 0:
          let msg = MessageDied(id: actor.id)
          pool.send(0, actor.parent_id, msg)
        

  #echo &"worker {worker.id} stopping"


proc getMyId*(c: Actor): ActorId {.cpsVoodoo.} =
  c.id

# Create pool with actor queue and worker threads

proc newPool*(nWorkers: int): ref Pool =

  var pool = new Pool
  initLock pool.workLock
  initCond pool.workCond

  for i in 0..<nWorkers:
    var worker = Worker(id: i) # Why the hell can't I initialize Worker(id: i, pool: Pool) ?
    worker.pool = pool[].addr
    pool.workers.add worker
    createThread(worker.thread, workerThread, worker)

  pool


# Wait until all actors in the pool have died and cleanup

proc run*(pool: ref Pool) =

  while true:
    withLock pool.mailhubLock:
      if pool.mailhubTable.len == 0:
        break
    os.sleep(50)

  echo "all mailboxes gone"

  withLock pool.workLock:
    pool.stop = true
    pool.workCond.broadcast()

  for worker in pool.workers:
    worker.thread.joinThread()

  echo "all workers stopped"




proc hatchAux(pool: ref Pool | ptr Pool, actor: sink Actor, parentId=0.ActorId): ActorId =

  pool.actorIdCounter += 1
  let myId = pool.actorIdCounter.load()

  actor.pool = pool[].addr
  actor.id = myId
  actor.parentId = parentId

  # Register a mailbox for the actor
  var mailbox = new Mailbox[Message]
  initLock mailbox.lock
  withLock pool.mailhubLock:
    pool.mailhubTable[actor.id] = mailbox

  # Add the new actor to the work queue
  withLock pool.workLock:
    pool.workQueue.addLast actor
    pool.workCond.signal()
    actor.wasMoved()

  myId

  
# Create and initialize a new actor

template hatch*(pool: ref Pool, c: typed): ActorId =
  # TODO verifyIsolated(actor)
  var actor = Actor(whelp c)
  hatchAux(pool, actor)


proc hatchFromActor*(actor: Actor, newActor: Actor): ActorId {.cpsVoodoo.} =
  let pool = actor.pool
  hatchAux(actor.pool, newActor, actor.id)

# Create and initialize a new actor from within an actor
#
template hatch*(c: typed): ActorId =
  var actor = Actor(whelp c)
  hatchFromActor(actor)


# Yield but go back to the work queue

proc backoff*(actor: sink Actor): Actor {.cpsMagic.} =
  let pool = actor.pool
  withLock pool.workLock:
    pool.workQueue.addLast(actor)
  actor.wasMoved()


