
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
import isisolated
import mallinfo


# FFI for glib mallinfo()

type 

  Pool* = object

    # Used to assign unique Actors
    actorPidCounter: Atomic[int]

    # All workers in the pool. No lock needed, only main thread touches this
    workers: seq[ref Worker]

    # This is where the continuations wait when not running
    workLock: Lock
    stop: bool
    workCond: Cond
    workQueue: Deque[ActorCond] # actor that needs to be run asap on any worker
    idleQueue: Table[Actor, ActorCond] # actor that is waiting for messages
    killReq: Table[Actor, bool]

    actorsLock: Lock
    actors: Table[Actor, bool]

  ActorCond* = ref object of Continuation
    actor*: Actor
    pool*: ptr Pool

  ActorObject* = object
    rc*: Atomic[int]
    pid*: int
    parent*: Actor

    lock: Lock
    links: seq[Actor]
    mailBox: Deque[Message]
    signalFd: cint

  Actor* = object
    p*: ptr ActorObject

  Worker = object
    id: int
    thread: Thread[ptr Worker]
    pool: ptr Pool

  ExitReason* = enum
    erNormal, erKilled, erError
  
  Message* = ref object of Rootobj
    src*: Actor

  MessageKill* = ref object of Message
  
  MessageExit* = ref object of Message
    id*: Actor
    reason*: ExitReason
    ex*: ref Exception

  MailFilter* = proc(msg: Message): bool


template log(a: Actor, msg: string) =
  #echo "\e[1;35m" & $a & ": " & msg & "\e[0m"
  discard


proc `=copy`*(dest: var Actor, ai: Actor) =
  if not ai.p.isNil:
    ai.p[].rc.atomicInc()
    ai.log("rc ++ " & $ai.p[].rc.load())
  doAssert dest.p.isNil
  dest.p = ai.p


proc `=destroy`*(ai: var Actor) =
  if not ai.p.isNil:
    if ai.p[].rc.load(moAcquire) == 0:
      ai.log("destroy")
      `=destroy`(ai.p[])
      deallocShared(ai.p)
    else:
      ai.p[].rc.atomicDec()
      ai.log("rc -- " & $ai.p[].rc.load())


proc `[]`*(ai: Actor): var ActorObject =
  assert not ai.p.isNil
  ai.p[]



proc newActor*(): Actor =
  result.p = create(ActorObject)
  #echo "ai: new ", cast[int](result.p)
  result.p[].rc.store(0)
  result.log("new")


template withLock(actor: Actor, code: untyped) =
  if not actor.p.isNil:
    withLock actor[].lock:
      code
  else:
    echo "empty actor"


# Forward declerations

proc kill*(pool: ptr Pool, id: Actor)


# Stringifications

proc `$`*(a: Actor): string =
  if not a.p.isNil:
    "actor." & $a.p[].pid
  else:
    "actor.nil"

proc `$`*(m: Message): string =
  if not m.isNil:
    return "msg"
  else:
    return "nil"

proc `$`*(pool: ref Pool): string =
  return "pool"

proc `$`*(a: ActorCond): string =
  return "actorcond." & $(a.actor.p[].pid)

proc `$`*(worker: ref Worker | ptr Worker): string =
  return "worker." & $worker.id


# Misc helper procs

proc pass*(cFrom, cTo: ActorCond): ActorCond =
  cTo.pool = cFrom.pool
  #cTo.id = cFrom.id
  cTo.actor = cFrom.actor
  cTo


# Send a message from src to dst

proc send*(pool: ptr Pool, src, dst: Actor, msg: sink Message) =
  assertIsolated(msg)
  #echo &"  send {src} -> {dst}: {msg.repr}"
  msg.src = src

  # Deliver the message in the target mailbox
  withLock dst:
    dst[].mailbox.addLast(msg)
    bitline.logValue("actor." & $dst & ".mailbox", dst[].mailbox.len)
    # If the target has a signalFd, wake it
    if dst[].signalFd != 0.cint:
      let b: char = 'x'
      discard posix.write(dst[].signalFd, b.addr, 1)

  # If the target continuation is in the sleep queue, move it to the work queue
  withLock pool.workLock:
    if dst in pool.idleQueue:
      #echo "wake ", dst
      let actor = pool.idleQueue[dst]
      pool.idleQueue.del(dst)
      pool.workQueue.addLast(actor)
      pool.workCond.signal()


# Receive a message

proc tryRecv*(pool: ptr Pool, actor: Actor, filter: MailFilter = nil): Message =
  withLock actor:
    var first = true
    for msg in actor[].mailbox.mitems:
      if not msg.isNil and (filter.isNil or filter(msg)):
        result = msg
        if first:
          actor[].mailbox.popFirst()
        else:
          msg = nil
        break
      first = false
  #echo &"  tryRecv {id}: {result}"


proc setSignalFd*(pool: ptr Pool, actor: Actor, fd: cint) =
  withLock actor:
    actor[].signalFd = fd

# Signal termination of an actor; inform the parent and kill any linked
# actors.

proc exit(c: sink ActorCond, reason: ExitReason, ex: ref Exception = nil) =
  #assertIsolated(c)  # TODO: cps refs child

  echo &"Actor {c.actor} terminated, reason: {reason}"
  if not ex.isNil:
    echo "Exception: ", ex.msg
    echo ex.getStackTrace()

  let pool = c.pool
  let actor = c.actor

  withLock actor:
  
    pool.send(Actor(), actor[].parent,
              MessageExit(id: c.actor, reason: reason, ex: ex))

    for id in actor[].links:
      {.cast(gcsafe).}:
        pool.kill(id)
    
    reset actor[].parent

  withLock pool.actorsLock:
    pool.actors.del(c.actor)


# Kill an actor

proc kill*(pool: ptr Pool, id: Actor) =
  withLock pool.workLock:
    # Mark the actor as to-be-killed so it will be caught before trampolining
    # or when jielding
    pool.killReq[id] = true
    # Send the actor a message so it will wake up if it is in the idle pool
  pool.send(Actor(), id, MessageKill())


# Move actor to the idle queue

proc toIdleQueue*(pool: ptr Pool, c: sink ActorCond) =
  #assertIsolated(c) # TODO
  var killed = false
  withLock pool.workLock:
    if c.actor in pool.killReq:
      pool.killReq.del(c.actor)
      killed = true
    else:
      pool.idleQueue[c.actor] = c

  if killed:
    exit(c, erKilled)


proc waitForWork(pool: ptr Pool): ActorCond =
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
            actor = actor.fn(actor).ActorCond
      except:
        actor.exit(erError, getCurrentException())

    # Cleanup if continuation has finished

    if actor.finished:
      actor.exit(erNormal)
      

proc hatchAux*(pool: ref Pool | ptr Pool, c: sink ActorCond, parent=Actor(), link=false): Actor =

  assert not isNil(c)
  assertIsolated(c)

  pool.actorPidCounter += 1
  let pid = pool.actorPidCounter.load()
  
  let actor = newActor()
  actor[].parent = parent
  actor[].lock.initLock()
  actor[].pid = pid

  # Initialize actor
  c.pool = pool[].addr
  c.actor = actor
  

  if link:
    actor[].links.add parent
  
  withLock pool.actorsLock:
    pool.actors[actor] = true

  # Add the new actor to the work queue
  withLock pool.workLock:
    assertIsolated(c)
    pool.workQueue.addLast c
    pool.workCond.signal()

  actor


# Create and initialize a new actor

template hatch*(pool: ref Pool, c: typed): Actor =
  var actor = ActorCond(whelp c)
  hatchAux(pool, actor)


# Create pool with actor queue and worker threads

proc newPool*(nWorkers: int): ref Pool =

  var pool = new Pool
  initLock pool.workLock
  initCond pool.workCond
  initLock pool.actorsLock

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
    withLock pool.actorsLock:
      if pool.actors.len == 0:
        break
      let mi = mallinfo2()
      bitline.logValue("stats.mailboxes", pool.actors.len)
      bitline.logValue("stats.mem_alloc", mi.uordblks)
      bitline.logValue("stats.mem_arena", mi.arena)
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


