
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

  Worker = ref object
    id: int
    thread: Thread[Worker]
    pool: ptr Pool

  Mailbox[T] = object
    lock: Lock
    queue: Deque[T]

  Pool = object

    # All workers in the pool. No lock needed, only main thread touches this
    workers: seq[Worker]
    
    workLock: Lock
    workCond: Cond
    workQueue: Deque[Work] # queue of awake work that needs to be run on any worker
    idleQueue: Table[string, Work] # here the idle work lie asleep

    # mailboxes for the workers
    mailhubLock: Lock
    mailhubTable: Table[string, ptr Mailbox[Message]]

  Message = ref object of Rootobj
    src: string



proc workerThread(worker: Worker) {.thread.} =
  let pool = worker.pool

  while true:

    echo &"worker {worker.id} loop"

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
        echo &"actor {work.id} has died"


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



proc hatchAux(pool: ref Pool, work: sink Work) =
  work.pool = pool[].addr

  # Register a mailbox for the actor
  let mailbox = create Mailbox[Message]
  initLock mailbox.lock
  withLock pool.mailhubLock:
    pool.mailhubTable[work.id] = mailbox

  # Add the new work to the work queue
  withLock pool.workLock:
    # TODO: verifyIsolated(work) ?
    pool.workQueue.addLast work
    pool.workCond.signal()
    work.wasMoved()



template hatch(pool: ref Pool, workId: string, c: typed) =
  # Create and initialize the new continuation
  var work = Work(whelp c)
  work.id = workId
  hatchAux(pool, work)


proc freeze(work: sink Work): Work {.cpsMagic.} =
  # If this continuation as a message waiting, move it back to the work queue.
  # Otherwise, put it in the sleep queue
  echo "freeze ", work.id
  let pool = work.pool
  withLock pool.mailhubLock:
    assert work.id in pool.mailhubTable
    let mailbox = pool.mailhubTable[work.id]
    withLock pool.workLock:
      if mailbox.queue.len == 0:
        echo "no mail for ", work.id
        pool.idleQueue[work.id] = work
      else:
        echo "mail for ", work.id
        pool.workQueue.addLast(work)
      work.wasMoved()


proc recvAux(work: Work): Message {.cpsVoodoo.} =
  let pool = work.pool
  withLock pool.mailhubLock:
    if work.id in pool.mailhubTable:
      let mailbox = pool.mailhubTable[work.id]
      withLock mailbox.lock:
        result = mailbox.queue.popFirst()
    else:
      raise ValueError.newException &"no mailbox for {work.id} found"


template recv(): Message =
  freeze()
  recvAux()


proc sendAux(work: Work, dstId: string, msg: sink Message): Work {.cpsMagic.} =

  msg.src = work.id

  # Find the mailbox for this actor
  let pool = work.pool
  withLock pool.mailhubLock:
    if work.id in pool.mailhubTable:
      let mailbox = pool.mailhubTable[dstId]
      # Deliver the message
      withLock mailbox.lock:
        mailbox.queue.addLast(msg)
      msg.wasMoved()
      # If the target continuation for is in the sleep queue, move it to the work queue
      withLock pool.workLock:
        if dstId in pool.idleQueue:
          echo "wake ", dstId
          var work = pool.idleQueue[dstId]
          pool.idleQueue.del(dstId)
          pool.workQueue.addLast(work)
          work.wasMoved
    else:
      raise ValueError.newException &"no mailbox for {dstId} found"
  work
      

template send(dstId: string, msg: Message): typed =
  echo "  -> ", dstId, ": ", msg.repr
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





