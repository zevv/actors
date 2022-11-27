
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
import mallinfo
import actorobj


# FFI for glib mallinfo()

type 

  Pool* = object

    # Used to assign unique Actors
    actorPidCounter: Atomic[int]
    actorCount: Atomic[int]

    # All workers in the pool. No lock needed, only main thread touches this
    workers: seq[ref Worker]

    # This is where the continuations wait when not running
    workLock: Lock
    stop: bool
    workCond: Cond
    workQueue: Deque[ActorCont] # actor that needs to be run asap on any worker
    idleQueue: Table[Actor, ActorCont] # actor that is waiting for messages
    killReq: Table[Actor, bool]

  ActorCont* = ref object of Continuation
    actor*: Actor
    pool*: ptr Pool

  Worker = object
    id: int
    thread: Thread[ptr Worker]
    pool: ptr Pool


# Forward declerations

proc kill*(pool: ptr Pool, id: Actor)


# Stringifications

proc `$`*(pool: ref Pool): string =
  return "pool"

proc `$`*(a: ActorCont): string =
  return "actorcond." & $(a.actor.p[].pid)

proc `$`*(worker: ref Worker | ptr Worker): string =
  return "worker." & $worker.id


# Misc helper procs

proc pass*(cFrom, cTo: ActorCont): ActorCont =
  cTo.pool = cFrom.pool
  #cTo.id = cFrom.id
  cTo.actor = cFrom.actor
  cTo


proc toWorkQueue*(pool: ptr Pool, actor: Actor) =
  # If the target continuation is in the sleep queue, move it to the work queue
  withLock pool.workLock:
    if actor in pool.idleQueue:
      #echo "wake ", dst
      let c = pool.idleQueue[actor]
      pool.idleQueue.del(actor)
      pool.workQueue.addLast(c)
      pool.workCond.signal()


# Send a message from src to dst

proc send*(pool: ptr Pool, src, dst: Actor, msg: sink Message) =
  dst.send(src, msg)
  pool.toWorkQueue(dst)


proc setSignalFd*(pool: ptr Pool, actor: Actor, fd: cint) =
  withLock actor:
    actor[].signalFd = fd


# Signal termination of an actor; inform the parent and kill any linked
# actors.

proc exit(pool: ptr Pool, actor: Actor, reason: ExitReason, ex: ref Exception = nil) =
  #assertIsolated(c)  # TODO: cps refs child

  echo &"Actor {actor} terminated, reason: {reason}"
  if not ex.isNil:
    echo "Exception: ", ex.msg
    echo ex.getStackTrace()

  var parent: Actor
  var links: seq[Actor]

  withLock actor:
    parent = move actor[].parent
    links = move actor[].links
  
  pool.send(Actor(), parent,
            MessageExit(id: actor, reason: reason, ex: ex))

  for id in links:
    {.cast(gcsafe).}:
      pool.kill(id)
    

  pool.actorCount -= 1


# Kill an actor

proc kill*(pool: ptr Pool, id: Actor) =
  withLock pool.workLock:
    # Mark the actor as to-be-killed so it will be caught before trampolining
    # or when jielding
    pool.killReq[id] = true
    # Send the actor a message so it will wake up if it is in the idle pool
  pool.send(Actor(), id, MessageKill())


# Move actor to the idle queue

proc toIdleQueue*(pool: ptr Pool, c: sink ActorCont) =
  #assertIsolated(c) # TODO
  var killed = false
  withLock pool.workLock:
    if c.actor in pool.killReq:
      pool.killReq.del(c.actor)
      killed = true
    else:
      pool.idleQueue[c.actor] = c

  if killed:
    pool.exit(c.actor, erKilled)


proc waitForWork(pool: ptr Pool): ActorCont =
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
    var c = pool.waitForWork()
    bitline.logStop(wid & ".wait")
    
    if c.isNil:
      break
    
    #assertIsolated(c)  # TODO: cps refs child

    # Trampoline the continuation

    bitline.log "worker." & $worker.id & ".run":
      try:
        {.cast(gcsafe).}: # Error: 'workerThread' is not GC-safe as it performs an indirect call here
          while not c.isNil and not c.fn.isNil:
            c = c.fn(c).ActorCont
      except:
        pool.exit(c.actor, erError, getCurrentException())

    # Cleanup if continuation has finished

    if c.finished:
      pool.exit(c.actor, erNormal)
      

proc hatchAux*(pool: ref Pool | ptr Pool, c: sink ActorCont, parent=Actor(), linked=false): Actor =

  assert not isNil(c)
  assertIsolated(c)

  pool.actorPidCounter += 1
  
  let actor = newActor()
  actor[].parent = parent
  actor[].lock.initLock()
  actor[].pid = pool.actorPidCounter.load()

  let a1 = actor
  let a2 = actor
  echo hash(a1)
  echo hash(a2)

  c.pool = pool[].addr
  c.actor = actor

  if linked:
    link(actor, parent)
 
  pool.actorCount += 1

  # Add the new actor to the work queue
  withLock pool.workLock:
    assertIsolated(c)
    pool.workQueue.addLast c
    pool.workCond.signal()

  actor


# Create and initialize a new actor

template hatch*(pool: ref Pool, c: typed): Actor =
  var actor = ActorCont(whelp c)
  hatchAux(pool, actor)


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

  while true:
    let n = pool.actorCount.load()
    let mi = mallinfo2()
    bitline.logValue("stats.actors", n)
    bitline.logValue("stats.mem_alloc", mi.uordblks)
    bitline.logValue("stats.mem_arena", mi.arena)
    if n == 0:
        break
    os.sleep(10)

  echo "all actors gone"

  withLock pool.workLock:
    pool.stop = true
    pool.workCond.broadcast()

  echo "waiting for workers to join"

  for worker in pool.workers:
    worker.thread.joinThread()
    assertIsolated(worker)

  echo "all workers stopped"


