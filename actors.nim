
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
import types
import mailbox
import isisolated


# FFI for glib mallinfo()

type 
  mallinfo = object
    arena: csize_t
    ordblks: csize_t
    smblks: csize_t
    hblks: csize_t
    hblkhd: csize_t
    usmblks: csize_t
    fsmblks: csize_t
    uordblks: csize_t
    fordblks: csize_t
    keepcost: csize_t

proc mallinfo2(): mallinfo {.importc: "mallinfo2".}


# Misc helper procs

proc `$`*(pool: ref Pool): string =
  return "#POOL<>"

proc `$`*(worker: ref Worker | ptr Worker): string =
  return "#WORKER<" & $worker.id & ">"

proc `$`*(a: Actor): string =
  return "#ACT<" & $a.parent_id & "." & $a.id & ">"

proc `$`*(m: Message): string =
  return "#MSG<" & $m.src & ">"

proc pass*(cFrom, cTo: Actor): Actor =
  cTo.pool = cFrom.pool
  cTo.id = cFrom.id
  cTo

macro actor*(n: untyped): untyped =
  n.addPragma nnkExprColonExpr.newTree(ident"cps", ident"Actor")
  n     


# Send a message from srcId to dstId

proc send*(pool: ptr Pool, srcId, dstId: ActorId, msg: sink Message) =

  msg.src = srcId
  #echo &"  send {srcId} -> {dstId}: {msg.repr}"

  pool.mailhub.withMailbox(dstId):
    assertIsolated(msg)
    mailbox.queue.addLast(msg)
    bitline.logValue("actor." & $dstId & ".mailbox", mailbox.queue.len)

  # If the target continuation is in the sleep queue, move it to the work queue
  withLock pool.workLock:
    if dstId in pool.idleQueue:
      #echo "wake ", dstId
      var actor = pool.idleQueue[dstId]
      pool.idleQueue.del(dstId)
      assertIsolated(actor)
      pool.workQueue.addLast(actor)
      pool.workCond.signal()

  # If the message is sent to the event queue, also write a byte to its
  # wake fd
  if dstId == pool.evqActorId:
    let b: char = 'x'
    discard posix.write(pool.evqFdWake, b.addr, 1)


proc sendAux*(actor: Actor, dst: ActorId, msg: sink Message) {.cpsVoodoo.} =
  actor.pool.send(actor.id, dst, msg)

template send*(dst: ActorId, msg: Message) =
  assertIsolated(msg)
  sendAux(dst, msg)


proc recvYield*(actor: sink Actor): Actor {.cpsMagic.} =
  # If there are no messages waiting in the mailbox, move the continuation to
  # the idle queue. Otherwise, return the current continuation so it can
  # receive and handle the mail without yielding
  let pool = actor.pool
  pool.mailhub.withMailbox(actor.id):
    if mailbox.queue.len == 0:
      withLock pool.workLock:
        pool.idleQueue[actor.id] = actor
        #actor = nil
    else:
      result = actor


proc recvGetMessage*(actor: Actor): Message {.cpsVoodoo.} =
  let pool = actor.pool
  pool.mailhub.withMailbox(actor.id):
    result = mailbox.queue.popFirst()
    assertIsolated(result)
    bitline.logValue("actor." & $actor.id & ".mailbox", mailbox.queue.len)


template recv*(): Message =
  recvYield()
  recvGetMessage()



proc waitForWork(pool: ptr Pool): Actor =
  withLock pool.workLock:
    while pool.workQueue.len == 0 and not pool.stop:
      pool.workCond.wait(pool.workLock)
    if not pool.stop:
      result = pool.workQueue.popFirst()
      assertIsolated(result)


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
    
    assertIsolated(actor)

    # Trampoline the continuation

    bitline.log "worker." & $worker.id & ".run":
      {.cast(gcsafe).}: # Error: 'workerThread' is not GC-safe as it performs an indirect call here
        actor = trampoline(actor)

    # Cleanup if continuation has finixhed

    if actor.finished:
      assertIsolated(actor)
      #echo &"actor {actor.id} has died, parent was {actor.parent_id}"
      pool.mailhub.unregister(actor.id)
      let msg = MessageDied(id: actor.id)
      pool.send(0, actor.parent_id, msg)
      



proc self*(c: Actor): ActorId {.cpsVoodoo.} =
  c.id

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

proc run*(pool: ref Pool) =

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

  for worker in pool.workers:
    worker.thread.joinThread()
    assertIsolated(worker)

  echo "all workers stopped"


proc hatchAux(pool: ref Pool | ptr Pool, actor: sink Actor, parentId=0.ActorId): ActorId =

  assert not isNil(actor)
  assertIsolated(actor)

  pool.actorIdCounter += 1
  let myId = pool.actorIdCounter.load()

  actor.pool = pool[].addr
  actor.id = myId
  actor.parentId = parentId

  # Register a mailbox for the actor
  pool.mailhub.register(actor.id)

  # Add the new actor to the work queue
  withLock pool.workLock:
    assertIsolated(actor)
    pool.workQueue.addLast actor
    pool.workCond.signal()

  myId

  
# Create and initialize a new actor

template hatch*(pool: ref Pool, c: typed): ActorId =
  var actor = Actor(whelp c)
  assertIsolated(actor)
  let id = hatchAux(pool, actor)
  actor = nil
  id

# Hatch an actor from within an actor

proc hatchFromActor*(actor: Actor, newActor: sink Actor): ActorId {.cpsVoodoo.} =
  assertIsolated(actor)
  hatchAux(actor.pool, newActor, actor.id)

# Create and initialize a new actor from within an actor

template hatch*(c: typed): ActorId =
  var actor = Actor(whelp c)
  assertIsolated(actor)
  # TODO: workaround for CPS problem: first assign to local var to prevent CPS
  # from moving the actor itself into the environment
  hatchFromActor(actor)

# Yield but go back to the work queue

proc backoff*(actor: sink Actor): Actor {.cpsMagic.} =
  let pool = actor.pool
  withLock pool.workLock:
    assertIsolated(actor)
    pool.workQueue.addLast(actor)


