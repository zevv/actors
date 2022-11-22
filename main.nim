
import os
import cps
import strformat
import isisolated
import std/locks
import std/deques
import std/tables


type

  Work = ref object of Continuation
    id: string
    pool: ptr Pool
    mailbox: Deque[Message]

  Worker = ref object
    id: int
    thread: Thread[Worker]
    pool: ptr Pool

  Pool = object

    # All workers in the pool. No lock needed, only main thread touches this
    workers: seq[Worker]
    
    # queue of awake actors that are scheduled to run on any worker.
    workLock: Lock
    workCond: Cond
    workQueue: Deque[Work]

    # list of all actors by id
    actorsLock: Lock
    actorsTable: Table[string, Work]

  Message = ref object of Rootobj
    src: string



proc workerThread(worker: Worker) {.thread.} =
  let pool = worker.pool

  while true:

    #echo &"worker {worker.id} loop"

    # Wait for work

    var work: Work
    withLock pool.workLock:
      while pool.workQueue.len == 0:
        pool.workCond.wait(pool.workLock)
      work = pool.workQueue.popFirst()

    # Trampoline once and push result back on the queue

    {.cast(gcsafe).}: # Error: 'workerThread' is not GC-safe as it performs an indirect call here

      if not work.fn.isNil:
        echo "tramp ", work.id
        discard trampoline(work)
      else:
        withLock pool.actorsLock:
          echo &"actor {work.id} has died"
          pool.actorsTable.del(work.id)


# Create pool with work queue and worker threads

proc newPool(nWorkers: int): ref Pool =

  var pool = new Pool
  initLock pool.workLock
  initCond pool.workCond

  for i in 0..<nWorkers:
    var worker = Worker(id: i) # Why the hell can't I ininitialze Worker(id: i, pool: Pool) ?
    worker.pool = pool[].addr
    pool.workers.add worker
    createThread(worker.thread, workerThread, worker)

  pool


proc run(p: ref Pool) =
  while true:
    os.sleep(50)


# Create a new actor and put it on the work queue

template hatch(pool: ref Pool, workId: string, c: typed) =

  # Create and initialize new work
  var work = Work(whelp c)
  work.pool = pool[].addr
  work.id = workId 

  # Add the new work to the actors table
  withLock pool.actorsLock:
    pool.actorsTable[work.id] = work

  # Add the new work to the work queue
  withLock pool.workLock:
    # TODO: verifyIsolated(work) ?
    pool.workQueue.addLast work
    pool.workCond.signal()
    wasmoved work


proc freeze(work: Work): Work {.cpsMagic.} =
  discard


proc recvAux(work: Work): Message {.cpsVoodoo.} =
  #echo &"recv {work.id}"
  work.mailbox.popFirst()


template recv(): Message =
  freeze()
  recvAux()


proc sendAux(work: Work, dstId: string, msg: Message): Work {.cpsMagic.} =

  msg.src = work.id

  # Find the actor in the actorTable
  let pool = work.pool
  withLock pool.actorsLock:
    if dstId in pool.actorsTable:
      let dstWork = pool.actorsTable[dstId]
      # Deliver the message
      dstWork.mailbox.addLast(msg)
      # Move the work to the awake work queue
      withLock pool.workLock:
        pool.workQueue.addLast dstWork
        pool.workCond.signal()
    else:
      raise ValueError.newException &"{dstId} does not exist"
  work
      

template send(dstId: string, msg: Message): typed =
  echo "  -> ", dstId, " :", msg.repr
  verifyIsolated(msg)
  sendAux(dstId, msg)


######################################################################

type

  MsgQuestion = ref object of Message
    a, b: int

  MsgAnswer = ref object of Message
    c: int

  MsgStop = ref object of Message


# This thing answers questions

proc alice() {.cps:Work.} =
  echo "I am alice"

  while true:
    let m = recv()

    if m of MsgQuestion:
      echo &"alice got a question from {m.src}"
      let mq = m.MsgQuestion
      send("bob", MsgAnswer(c: mq.a + mq.b))

    if m of MsgStop:
      echo "alice says bye"
      break


proc bob() {.cps:Work.} =
  echo "I am bob"


  var i = 0

  while i < 10:
    # Let's ask alice a question
    
    send("alice", MsgQuestion(a: 10, b: i))

    # And receive the answer
    let m = recv()

    if m of MsgAnswer:
      let ma = m.MsgAnswer
      echo &"bob received an answer from {ma.src}: {ma.c}"

    inc i

  # Thank you alice, you can go now

  send("alice", MsgStop())



var pool = newPool(2)

pool.hatch "alice", alice()
pool.hatch "bob", bob()

pool.run()





