
#
#
#
#
#
#
#
#
#
#
import os
import std/macros
import std/locks
import std/deques
import std/strformat
import std/tables
import std/posix
import std/atomics
import std/times
import std/hashes
import std/posix

import cps

import bitline
import isisolated
import mallinfo


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
    workQueue: Deque[Actor] # actor that needs to be run asap on any worker

  ActorCont* = ref object of Continuation
    actor*: Actor
    pool*: ptr Pool

  Worker = object
    id: int
    thread: Thread[ptr Worker]
    pool: ptr Pool

  State = enum
    Suspended, Queued, Running, Killed, Dead

  ActorObject* = object
    rc*: Atomic[int]
    pool*: ptr Pool
    pid*: int
    parent*: Actor
    msgQueue*: Deque[Message]
    links*: seq[Actor]

    lock*: Lock
    state*: State
    sigQueue*: Deque[Signal]
    c*: Continuation
    signalFd*: cint

  Actor* = object
    p*: ptr ActorObject

  Signal* = ref object of Rootobj
    src*: Actor

  SigKill = ref object of Signal

  SigLink = ref object of Signal

  Message* = ref object of Signal

  MessageExit* = ref object of Message
    actor*: Actor
    reason*: ExitReason
    ex*: ref Exception

  ExitReason* = enum
    Normal, Killed, Error

  MailFilter* = proc(msg: Message): bool



# `Actor` is a custom atomic RC type

proc `=destroy`*(actor: var Actor)

proc `=copy`*(dest: var Actor, actor: Actor) =
  if not actor.p.isNil:
    actor.p[].rc.atomicInc()
  if not dest.p.isNil:
    `=destroy`(dest)
  dest.p = actor.p


proc `=destroy`*(actor: var Actor) =
  if not actor.p.isNil:
    if actor.p[].rc.load(moAcquire) == 0:
      `=destroy`(actor.p[])
      deallocShared(actor.p)
    else:
      actor.p[].rc.atomicDec()


proc `[]`*(actor: Actor): var ActorObject =
  assert not actor.p.isNil
  actor.p[]


proc isNil*(actor: Actor): bool =
  actor.p.isNil




# Stringifications

proc `$`*(pool: ptr Pool): string =
  return "pool"

proc `$`*(c: ActorCont): string =
  if c.isNil: "actorcont.nil" else: "actorcont." & $c.actor

proc `$`*(worker: ref Worker | ptr Worker): string =
  return "worker." & $worker.id

proc `$`*(a: Actor): string =
  result = "actor." & (if a.isNil: "nil" else: $a.p[].pid)

proc `$`*(m: Signal): string =
  if not m.isNil: "sig src=" & $m.src else: "sig.nil"

proc `$`*(m: Message): string =
  if not m.isNil: "msg src=" & $m.src else: "msg.nil"


# CPS `pass` implementation

proc pass*(cFrom, cTo: ActorCont): ActorCont =
  #echo &"pass #{cast[int](cFrom[].addr):#x} #{cast[int](cTo[].addr):#x}"
  cTo.pool = cFrom.pool
  cTo.actor = cFrom.actor
  cTo


# handle incoming signal queue

proc handleSignals(actor: Actor) =

  while true:

    var sig: Signal
    withLock actor[].lock:
      if actor[].sigQueue.len() == 0:
        break
      sig = actor[].sigQueue.popFirst()

    if sig of Message:
      actor[].msgQueue.addLast(sig.Message)

    elif sig of SigKill:
      actor[].state = Killed

    elif sig of SigLink:
      actor[].links.add sig.src

    else:
      echo actor, ": rx unknown signal"


# Move the continuation back into the actor; the actor can later be resumed to
# continue the continuation. If there are signals waiting, handle those and do
# not suspend.

proc trySuspend*(actor: Actor, c: sink Continuation): bool =

  withLock actor[].lock:
    if actor[].sigQueue.len == 0:
      actor[].c = move c
      if actor[].state == Running:
        actor[].state = Suspended
      return true

  actor.handleSignals()
  return false


# Move the actors continuation to the work queue to schedule execution
# on the worker threads

proc resume*(pool: ptr Pool, actor: Actor) =
  var fd: cint
  withLock pool.workLock:
    withLock actor[].lock:
      if actor[].state == Suspended:
        actor[].state = Queued
        pool.workQueue.addLast(actor)
        pool.workCond.signal()
      fd = actor[].signalFd

  if fd != 0.cint:
    let b = 'x'
    discard posix.write(fd, b.addr, sizeof(b))


# Move a running process to the back of the work queue

proc jield*(actor: Actor, c: sink ActorCont) =
  actor.handleSignals()
  let pool = c.pool
  withLock pool.workLock:
    withLock actor[].lock:
      doAssert actor[].state == Running
      if actor[].state != Killed:
        actor[].state = Queued
        actor[].c = move c
        pool.workQueue.addLast(actor)
        pool.workCond.signal()


# Set signal file descriptor

proc setSignalFd*(actor: Actor, fd: cint) =
  withLock actor[].lock:
    actor[].signalFd = fd


# Send a message from src to dst

proc send*(actor: Actor, sig: sink Signal, src: Actor) =
  #echo "  ", src, " -> ", actor
  sig.src = src

  withLock actor[].lock:
    if actor[].state notin {Killed, Dead}:
      actor[].sigQueue.addLast(sig)
  
  actor[].pool.resume(actor)


# Link two processes: if one goes down, the other gets killed as well

proc link*(actor: Actor, peer: Actor) =
  actor[].links.add(peer)
  peer.send(SigLink(), actor)


# Kill an actor

proc kill*(actor: Actor) =
  actor.send(SigKill(), Actor())


# Signal termination of an actor; inform the parent and kill any linked
# actors.

proc exit(pool: ptr Pool, actor: Actor, reason: ExitReason, ex: ref Exception = nil) =

  withLock actor[].lock:
    actor[].state = Dead
  
  actor.handleSignals()

  # Inform the parent of the death of their child
  if not actor[].parent.isnil:
    # getCurrentException() returns a ref to a global .threadvar, which is not
    # safe to carry around. As a workaround, create a fresh exception and copy
    # some of the fields. TODO: what about the stack frames?
    var ex: ref Exception
    let exCur = getCurrentException()
    if not exCur.isNil:
      ex = new Exception
      ex.name = exCur.name
      ex.msg = exCur.msg
    actor[].parent.send(MessageExit(actor: actor, reason: reason, ex: ex), Actor())
  else:
    # This actor has no parent, dump termination info to console
    echo &"Actor {actor} has terminated, reason: {reason}"
    if reason == Error:
      let ex = getCurrentException()
      echo "Exception: ", ex.name, ": ", ex.msg
      echo getStackTrace(ex)
      quit 1
  
  # Kill linked processes
  for link in actor[].links:
    kill(link)

  withLock actor[].lock:
    actor[].sigQueue.clear()
    # Drop the continuation
    if not actor[].c.isNil:
      actor[].c.disarm
      actor[].c = nil

  actor[].msgQueue.clear()
  actor[].links.setLen(0)
  actor[].parent = Actor()

  pool.actorCount -= 1


# Thread main function: receives actors from the work queue, trampolines
# their continuations and handles exit conditions

proc workerThread(worker: ptr Worker) {.thread.} =

  let pool = worker.pool
  let wid = "worker." & $worker.id

  while true:

    # Wait for the next actor to be available in the work queue
    var actor: Actor
    var c: Continuation

    withLock pool.workLock:
      while pool.workQueue.len == 0 and not pool.stop:
        pool.workCond.wait(pool.workLock)
      if pool.stop:
        break
      actor = pool.workQueue.popFirst()
      withLock actor[].lock:
        if actor[].state == Dead:
          continue
        doAssert actor[].state == Queued
        actor[].state = Running
        c = move actor[].c

    actor.handleSignals()

    try:

      # Trampoline the actor's continuation if in the Running state

      {.cast(gcsafe).}:
        while not c.isNil and not c.fn.isNil:
          c = c.fn(c)

    except:
      pool.exit(actor, Error)
        

    # Cleanup if continuation has finished or was killed
    if c.finished:
      pool.exit(actor, Normal)
    else:
      var state: State
      withLock actor[].lock:
        state = actor[].state
      if state == Killed:
        pool.exit(actor, Killed)
    

# Try to receive a message, returns `nil` if no messages available or matched
# the passed filter

proc tryRecv*(actor: Actor, filter: MailFilter = nil): Message =
  var first = true
  for msg in actor[].msgQueue.mitems:
    if not msg.isNil and (filter.isNil or filter(msg)):
      result = msg
      if first:
        actor[].msgQueue.popFirst()
      else:
        msg = nil
      break
    first = false


proc hatchAux*(pool: ptr Pool, c: sink ActorCont, parent=Actor(), linked=false): Actor =

  assert not isNil(c)
  #assertIsolated(c)

  let a = create(ActorObject)
  a.pid = pool.actorPidCounter.fetchAdd(1)
  a.pool = pool
  a.rc.store(0)
  a.lock.initLock()
  a.parent = parent
  var actor = Actor()
  actor.p = a

  c.pool = pool
  c.actor = actor
  actor[].c = move c

  if linked:
    link(actor, parent)

  pool.actorCount += 1
  pool.resume(actor)

  actor


# Create pool with actor queue and worker threads

proc newPool*(nWorkers: int): ref Pool =

  var pool = new Pool
  initLock pool.workLock
  initCond pool.workCond
  pool.actorPidCounter.store(1)

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

  withLock pool.workLock:
    pool.stop = true
    pool.workCond.broadcast()

  for worker in pool.workers:
    worker.thread.joinThread()
    assertIsolated(worker)



