
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
    Active, Killed, Dead

  ActorObject* = object
    rc*: Atomic[int]
    pool*: ptr Pool
    pid*: int
    parent*: Actor

    lock*: Lock
    state*: State
    c*: Continuation
    links*: seq[Actor]
    mailBox*: Deque[Message]
    signalFd*: cint

  Actor* = object
    p*: ptr ActorObject

  Message* = ref object of Rootobj
    src*: Actor

  MessageKill* = ref object of Message
  
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


proc isRunning(actor: Actor): bool = actor[].c.isNil
proc isSuspended(actor: Actor): bool = not actor.isRunning()


# Stringifications

proc `$`*(pool: ptr Pool): string =
  return "pool"

proc `$`*(c: ActorCont): string =
  if c.isNil: "actorcont.nil" else: "actorcont." & $c.actor

proc `$`*(worker: ref Worker | ptr Worker): string =
  return "worker." & $worker.id

proc `$`*(a: Actor): string =
  result = "actor." & (if a.isNil: "nil" else: $a.p[].pid)

proc `$`*(m: Message): string =
  if not m.isNil: "msg src=" & $m.src else: "msg.nil"


# CPS `pass` implementation

proc pass*(cFrom, cTo: ActorCont): ActorCont =
  #echo &"pass #{cast[int](cFrom[].addr):#x} #{cast[int](cTo[].addr):#x}"
  cTo.pool = cFrom.pool
  cTo.actor = cFrom.actor
  cTo


# Link two processes: if one goes down, the other gets killed as well

proc link*(a: Actor, b: Actor) =
  withLock a[].lock:
    a[].links.add b
  withLock b[].lock:
    b[].links.add a


# Move the continuation back into the actor; the actor can later be
# resumed to continue the continuation

proc suspend*(actor: Actor, c: sink Continuation) =
  withLock actor[].lock:
    doAssert actor.isRunning()
    actor[].c = move c


# Move the actors continuation to the work queue to schedule execution
# on the worker threads

proc resume*(pool: ptr Pool, actor: Actor) =
  var fd: cint
  withLock pool.workLock:
    withLock actor[].lock:
      if actor.isSuspended():
        pool.workQueue.addLast(actor)
        pool.workCond.signal()
      fd = actor[].signalFd

  if fd != 0.cint:
    let b = 'x'
    discard posix.write(fd, b.addr, sizeof(b))


# Move a running process to the back of the work queue

proc jield*(actor: Actor, c: sink ActorCont) =
  let pool = c.pool
  withLock pool.workLock:
    withLock actor[].lock:
      doAssert actor.isRunning()
      if actor[].state != Killed:
        actor[].c = move c
        pool.workQueue.addLast(actor)
        pool.workCond.signal()


# Set signal file descriptor

proc setSignalFd*(actor: Actor, fd: cint) =
  withLock actor[].lock:
    actor[].signalFd = fd


# Send a message from src to dst

proc send*(actor: Actor, msg: sink Message, src: Actor) =

  #echo "  ", src, " -> ", actor

  msg.src = src

  var count: int
  withLock actor[].lock:
    actor[].mailbox.addLast(msg)

  actor[].pool.resume(actor)


# Kill an actor

proc kill*(actor: Actor) =
  withLock actor[].lock:
    if actor[].state == Active:
      actor[].state = Killed
  actor.send(MessageKill(), Actor())


# Signal termination of an actor; inform the parent and kill any linked
# actors.

proc exit(pool: ptr Pool, actor: Actor, reason: ExitReason, ex: ref Exception = nil) =

  #echo &"Actor {actor} terminated, reason: {reason}"

  var parent: Actor
  var links: seq[Actor]

  withLock actor[].lock:
    actor[].state = Dead
    actor[].mailbox.clear()
    parent = move actor[].parent
    links = move actor[].links

  # Inform the parent of the death of their child
  if not parent.isnil:
    parent.send(MessageExit(actor: actor, reason: reason, ex: ex), Actor())

  # Drop the continuation
  if actor.isSuspended():
    actor[].c.disarm
    actor[].c = nil

  # Kill linked processes
  for id in links:
    kill(id)
  
  pool.actorCount -= 1


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
        c = move actor[].c
        if actor[].state == Dead:
          continue

    try:
    
      # Trampoline the actor's continuation if in the Running state

      {.cast(gcsafe).}:
        while not c.isNil and not c.fn.isNil:
          c = c.fn(c)

    except:
      # getCurrentException() returns a ref to a global .threadvar, which is
      # not safe to carry around. As a workaround we create a fresh
      # exception and copy some of the fields # TODO what about the stack
      # frames?
      let ex = getCurrentException()
      let exNew = new Exception
      exNew.name = ex.name
      exNew.msg = ex.msg
      pool.exit(actor, Error, exNew)

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
  withLock actor[].lock:
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

  echo "joining workers"
  withLock pool.workLock:
    pool.stop = true
    pool.workCond.broadcast()

  for worker in pool.workers:
    worker.thread.joinThread()
    assertIsolated(worker)



