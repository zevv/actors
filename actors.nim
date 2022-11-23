
import os
import cps
import strformat
import isisolated
import std/locks
import std/deques
import std/tables
import std/atomics


type

  ActorId* = int

  Work* = ref object of Continuation
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

    actorIdCounter: Atomic[int]

    # All workers in the pool. No lock needed, only main thread touches this
    workers: seq[Worker]
   
    # This is where the continuations wait when not running
    workLock: Lock
    stop: bool
    workCond: Cond
    workQueue: Deque[Work] # work that needs to be run asap on any worker
    idleQueue: Table[ActorId, Work] # work that is waiting for messages

    # mailboxes for the actors
    mailhubLock: Lock
    mailhubTable: Table[ActorId, Mailbox[Message]]

  Message* = ref object of Rootobj
    src*: ActorId


proc pass*(cFrom, cTo: Work): Work =
  cTo.pool = cFrom.pool
  cTo.id = cFrom.id
  cTo


proc workerThread(worker: Worker) {.thread.} =
  let pool {.cursor.} = worker.pool

  while true:

    # Wait for work or stop request

    var work: Work
    withLock pool.workLock:
      while pool.workQueue.len == 0 and not pool.stop:
        pool.workCond.wait(pool.workLock)
      if pool.stop:
        break
      work = pool.workQueue.popFirst()

    # Trampoline once and push result back on the queue

    if not work.fn.isNil:
      echo "\e[35mtramp ", work.id, " on worker ", worker.id, "\e[0m"
      {.cast(gcsafe).}: # Error: 'workerThread' is not GC-safe as it performs an indirect call here
        work = trampoline(work)
      if not isNil(work) and isNil(work.fn):
        echo &"actor {work.id} has died"
        # Delete the mailbox for this actor
        withLock pool.mailhubLock:
          pool.mailhubTable.del(work.id)

  #echo &"worker {worker.id} stopping"


proc getMyId*(c: Work): ActorId {.cpsVoodoo.} =
  c.id

# Create pool with work queue and worker threads

proc newPool*(nWorkers: int): Pool =

  var pool = new Pool
  initLock pool.workLock
  initCond pool.workCond

  for i in 0..<nWorkers:
    var worker = Worker(id: i) # Why the hell can't I ininitialze Worker(id: i, pool: Pool) ?
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



proc hatchAux(pool: Pool, work: sink Work): ActorId =

  pool.actorIdCounter += 1
  let myId = pool.actorIdCounter.load()

  work.pool = pool
  work.id = myId

  # Register a mailbox for the actor
  var mailbox = Mailbox[Message]()
  initLock mailbox.lock
  withLock pool.mailhubLock:
    pool.mailhubTable[work.id] = mailbox

  # Add the new work to the work queue
  withLock pool.workLock:
    pool.workQueue.addLast work
    pool.workCond.signal()
    work.wasMoved()

  myId

  
# Create and initialize a new actor

template hatch*(pool: Pool, c: typed): ActorId =
  # TODO verifyIsolated(work)
  var work = Work(whelp c)
  hatchAux(pool, work)


proc freeze*(work: sink Work): Work {.cpsMagic.} =
  # If this continuation as a message waiting, move it back to the work queue.
  # Otherwise, put it in the sleep queue
  #echo "freeze ", work.id
  let pool {.cursor.} = work.pool
  withLock pool.mailhubLock:
    assert work.id in pool.mailhubTable
    let mailbox {.cursor.} = pool.mailhubTable[work.id]
    withLock pool.workLock:
      if mailbox.queue.len == 0:
        #echo "no mail for ", work.id
        pool.idleQueue[work.id] = work
      else:
        #echo "mail for ", work.id
        pool.workQueue.addLast(work)
      work.wasMoved()


proc recvAux*(work: Work): Message {.cpsVoodoo.} =
  let pool {.cursor.} = work.pool
  withLock pool.mailhubLock:
    if work.id in pool.mailhubTable:
      let mailbox = pool.mailhubTable[work.id]
      withLock mailbox.lock:
        result = mailbox.queue.popFirst()
    else:
      raise ValueError.newException &"no mailbox for {work.id} found"


template recv*(): Message =
  freeze()
  recvAux()


proc sendAux*(work: Work, dstActorId: ActorId, msg: sink Message): Work {.cpsMagic.} =

  msg.src = work.id

  # Find the mailbox for this actor
  let pool {.cursor.} = work.pool
  withLock pool.mailhubLock:
    if work.id in pool.mailhubTable:
      let mailbox = pool.mailhubTable[dstActorId]
      # Deliver the message
      withLock mailbox.lock:
        mailbox.queue.addLast(msg)
      msg.wasMoved()
      # If the target continuation is in the sleep queue, move it to the work queue
      withLock pool.workLock:
        if dstActorId in pool.idleQueue:
          #echo "wake ", dstActorId
          var work = pool.idleQueue[dstActorId]
          pool.idleQueue.del(dstActorId)
          pool.workQueue.addLast(work)
          work.wasMoved
    else:
      raise ValueError.newException &"no mailbox for {dstActorId} found"
  work
      

template send*(dstActorId: ActorId, msg: Message): typed =
  echo "  -> ", dstActorId, ": ", msg.repr
  verifyIsolated(msg)
  sendAux(dstActorId, msg)


