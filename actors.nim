
import os
import cps
import strformat
import isisolated
import std/locks
import std/deques
import std/tables
import std/atomics
import std/times


const optLog = false

type

  ActorId* = int

  Actor* = ref object of Continuation
    id: ActorId
    pool: Pool

  Worker = ref object
    id: int
    thread: Thread[Worker]
    pool: Pool

  Mailbox[T] = ref object
    lock: Lock
    queue: Deque[T]

  Pool* = ref object

    # bitline log file
    fLog: File

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


proc pass*(cFrom, cTo: Actor): Actor =
  cTo.pool = cFrom.pool
  cTo.id = cFrom.id
  cTo


proc log(pool: Pool, event, msg: string) =
  when optLog:
    let l = $epochTime() & " " & event & " " & msg & "\n"
    pool.fLog.write(l)
    pool.fLog.flushFile


proc workerThread(worker: Worker) {.thread.} =
  let pool {.cursor.} = worker.pool
  let wid = "worker." & $worker.id

  while true:

    # Wait for actor or stop request

    pool.log("+", wid & ".wait\n")

    var actor: Actor
    withLock pool.workLock:
      while pool.workQueue.len == 0 and not pool.stop:
        pool.workCond.wait(pool.workLock)
      if pool.stop:
        break
      actor = pool.workQueue.popFirst()

    pool.log("-", wid & ".wait\n")

    # Trampoline once and push result back on the queue

    if not actor.fn.isNil:
      echo "\e[35mtramp ", actor.id, " on worker ", worker.id, "\e[0m"
      {.cast(gcsafe).}: # Error: 'workerThread' is not GC-safe as it performs an indirect call here
        let aid = "actor." & $actor.id
        let wid = "worker." & $worker.id

        pool.log("+", wid & ".run")
        pool.log("+", aid & ".run")

        actor = trampoline(actor)

        pool.log("-", wid & ".run")
        pool.log("-", aid & ".run")

      if not isNil(actor) and isNil(actor.fn):
        echo &"actor {actor.id} has died"
        # Delete the mailbox for this actor
        withLock pool.mailhubLock:
          pool.mailhubTable.del(actor.id)

  #echo &"worker {worker.id} stopping"


proc getMyId*(c: Actor): ActorId {.cpsVoodoo.} =
  c.id

# Create pool with actor queue and worker threads

proc newPool*(nWorkers: int): Pool =

  var pool = new Pool
  initLock pool.workLock
  initCond pool.workCond

  when optLog:
    pool.fLog = open("/tmp/nimactors.bl", fmWrite)

  for i in 0..<nWorkers:
    var worker = Worker(id: i) # Why the hell can't I initialize Worker(id: i, pool: Pool) ?
    worker.pool = pool
    pool.workers.add worker
    createThread(worker.thread, workerThread, worker)

  pool


# Wait until all actors in the pool have died and cleanup

proc run*(pool: Pool) =

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

  when optLog:
    poll.fdLog.close()




proc hatchAux(pool: Pool, actor: sink Actor): ActorId =

  pool.actorIdCounter += 1
  let myId = pool.actorIdCounter.load()

  actor.pool = pool
  actor.id = myId

  # Register a mailbox for the actor
  var mailbox = Mailbox[Message]()
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

template hatch*(pool: Pool, c: typed): ActorId =
  # TODO verifyIsolated(actor)
  var actor = Actor(whelp c)
  hatchAux(pool, actor)


proc freeze*(actor: sink Actor): Actor {.cpsMagic.} =
  # If this continuation as a message waiting, move it back to the work queue.
  # Otherwise, put it in the sleep queue
  #echo "freeze ", work.id
  let pool {.cursor.} = actor.pool
  withLock pool.mailhubLock:
    assert actor.id in pool.mailhubTable
    let mailbox {.cursor.} = pool.mailhubTable[actor.id]
    withLock pool.workLock:
      if mailbox.queue.len == 0:
        #echo "no mail for ", actor.id
        pool.idleQueue[actor.id] = actor
      else:
        #echo "mail for ", actor.id
        pool.workQueue.addLast(actor)
      actor.wasMoved()


proc recvAux*(actor: Actor): Message {.cpsVoodoo.} =
  let pool {.cursor.} = actor.pool
  withLock pool.mailhubLock:
    if actor.id in pool.mailhubTable:
      let mailbox = pool.mailhubTable[actor.id]
      withLock mailbox.lock:
        result = mailbox.queue.popFirst()
    else:
      raise ValueError.newException &"no mailbox for {actor.id} found"


template recv*(): Message =
  freeze()
  recvAux()


proc sendAux*(actor: Actor, dstActorId: ActorId, msg: sink Message): Actor {.cpsMagic.} =

  msg.src = actor.id

  # Find the mailbox for this actor
  let pool {.cursor.} = actor.pool
  withLock pool.mailhubLock:
    if actor.id in pool.mailhubTable:
      let mailbox = pool.mailhubTable[dstActorId]
      # Deliver the message
      withLock mailbox.lock:
        mailbox.queue.addLast(msg)
      #msg.wasMoved()   # really, was it?
      # If the target continuation is in the sleep queue, move it to the work queue
      withLock pool.workLock:
        if dstActorId in pool.idleQueue:
          #echo "wake ", dstActorId
          var actor = pool.idleQueue[dstActorId]
          pool.idleQueue.del(dstActorId)
          pool.workQueue.addLast(actor)
          pool.workCond.signal()
          pool.log("!", "actor." & $actor.id & ".signal")
          actor.wasMoved
    else:
      raise ValueError.newException &"no mailbox for {dstActorId} found"
  actor


template send*(dstActorId: ActorId, msg: Message): typed =
  echo "  -> ", dstActorId, ": ", msg.repr
  verifyIsolated(msg)
  sendAux(dstActorId, msg)


