import std/[os, osproc, macros, locks, deques, strformat, tables, atomics, times] 
import cps

import bitline
import isisolated
import mallinfo
import valgrind

{.emit:"#include <valgrind/helgrind.h>".}

type

  Pool* = object

    # Used to assign unique Actors
    actorPidCounter: Atomic[int]
    actorCount: Atomic[int]

    # All workers in the pool. No lock needed, only main thread touches this
    workers: seq[ref Worker]

    # This is where the continuations wait when not running
    lock: Lock
    stop {.guard:lock.}: bool
    cond {.guard:lock.}: Cond
    workQueue {.guard:lock.}: Deque[Actor] # actor that needs to be run asap on any worker

  ActorCont* = ref object of Continuation
    actor*: Actor
    pool*: ptr Pool

  Worker = object
    id: int
    thread: Thread[ptr Worker]
    pool: ptr Pool

  State = enum
    Suspended, Queued, Running, Suspending, Jielding, Killed, Dead

  ActorObject* = object
    rc*: Atomic[int]
    pool*: ptr Pool
    pid*: int
    msgQueue*: Deque[Message]
    monitors*: seq[Actor]
    links*: seq[Actor]

    lock*: Lock
    state* {.guard:lock.}: State
    sigQueue* {.guard:lock.}: Deque[Signal]
    c* {.guard:lock.}: Continuation

  Actor* = object
    p*: ptr ActorObject

  Signal* = ref object of Rootobj
    src*: Actor

  SigKill = ref object of Signal

  SigLink = ref object of Signal

  SigMonitor = ref object of Signal

  Message* = ref object of Signal

  MessageExit* = ref object of Message
    actor*: Actor
    reason*: ExitReason
    ex*: ref Exception

  ExitReason* = enum
    Normal, Killed, Error, Lost

  MailFilter* = proc(msg: Message): bool


proc exit(pool: ptr Pool, actor: Actor, reason: ExitReason, ex: ref Exception = nil)


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
    if actor.p[].rc.fetchSub(1) == 0:
      valgrind_annotate_happens_after(actor.p[].rc.addr)
      valgrind_annotate_happens_before_forget_all(actor.p[].rc.addr)
      actor.p.pool.exit(actor, Lost)
      `=destroy`(actor.p[])
      deallocShared(actor.p)
      actor.p = nil
    else:
      valgrind_annotate_happens_before(actor.p[].rc.addr)

proc `[]`*(actor: Actor): var ActorObject =
  assert not actor.p.isNil
  actor.p[]

proc isNil*(actor: Actor): bool =
  actor.p.isNil


# Stringifications

proc `$`*(pool: ptr Pool): string =
  return "pool"

proc `$`*(worker: ref Worker | ptr Worker): string =
  return "worker." & $worker.id

proc `$`*(a: Actor): string =
  result = "actor." & (if a.isNil: "nil" else: $a.p[].pid)

proc `$`*(m: Signal): string =
  if not m.isNil: "sig src=" & $m.src else: "sig.nil"

proc `$`*(m: Message): string =
  if not m.isNil: "msg src=" & $m.src else: "msg.nil"

proc `$`*(c: ActorCont): string =
  if c.isNil: "actorcont.nil" else: "actorcont." & $c.actor


# CPS `pass` implementation

proc pass*(cFrom, cTo: ActorCont): ActorCont =
  #echo &"pass #{cast[int](cFrom[].addr):#x} #{cast[int](cTo[].addr):#x}"
  cTo.pool = cFrom.pool
  cTo.actor = cFrom.actor
  cTo


# locking templates

template withLock(pool: ptr Pool or ref Pool, code: typed) =
  withLock pool.lock:
    {.locks: [pool.lock].}:
      code

template withLock(actor: Actor, code: typed) =
  withLock actor[].lock:
    {.locks: [actor[].lock].}:
      code

# handle incoming signal queue

proc handleSignals(actor: Actor) =

  while true:

    var sig: Signal
    actor.withLock:
      if actor[].sigQueue.len() == 0:
        break
      sig = actor[].sigQueue.popFirst()

    if sig of Message:
      actor[].msgQueue.addLast(sig.Message)

    elif sig of SigKill:
      actor.withLock:
        actor[].state = Killed

    elif sig of SigLink:
      actor[].links.add sig.src
    
    elif sig of SigMonitor:
      echo actor, " monitored by ", sig.src
      actor[].monitors.add sig.src

    else:
      echo actor, ": rx unknown signal"


# Move the continuation back into the actor; the actor can later be resumed to
# continue the continuation. If there are signals waiting, handle those and do
# not suspend.

proc suspend*(actor: Actor, c: sink Continuation): ActorCont =

  actor.withLock:
    if actor[].sigQueue.len == 0:
      actor[].c = move c
      if actor[].state == Running:
        actor[].state = Suspending
      return nil

  actor.handleSignals()
  return c.ActorCont


# Move the actors continuation to the work queue to schedule execution
# on the worker threads

proc resume*(pool: ptr Pool, actor: Actor) =
  pool.withLock:
    actor.withLock:
      if actor[].state == Suspended:
        actor[].state = Queued
        pool.workQueue.addLast(actor)
        pool.cond.signal()


# Indicate the actor has yielded

proc jield*(actor: Actor, c: sink ActorCont) =
  actor.handleSignals()
  actor.withLock:
    if actor[].state == Running:
      actor[].c = move c
      actor[].state = Jielding
    else:
      doAssert actor[].state == Killed


# Send a message from src to dst

proc sendSig*(actor: Actor, sig: sink Signal, src: Actor) =
  #echo "  ", src, " -> ", actor
  sig.src = src

  actor.withLock:
    if actor[].state notin {State.Killed, State.Dead}:
      actor[].sigQueue.addLast(move sig)

  actor[].pool.resume(actor)


# Get the message from the actor's message queue at index `idx`

proc getMsg*(actor: Actor, idxStart: int): tuple[msg: Message, idxNext: int] =
  let len = actor[].msgQueue.len
  var idx = idxStart
  while idx < len:
    if not actor[].msgQueue[idx].isnil:
      result = (actor[].msgQueue[idx], idx)
      break
    else:
      inc idx


# Drop the message from the actor's message queue at index `idx`

proc dropMsg*(actor: Actor, idx: Natural) =
  let len = actor[].msgQueue.len
  if idx == 0:
    actor[].msgQueue.shrink(1, 0)
  elif idx == len - 1:
    actor[].msgQueue.shrink(0, 1)
  else:
    actor[].msgQueue[idx] = nil


# Link two processes: if one goes down, the other gets killed as well

proc link*(actor: Actor, peer: Actor) =
  actor[].links.add(peer)
  peer.sendSig(SigLink(), actor)


# Monitor a process

proc monitor*(actor: Actor, peer: Actor) =
  peer.sendSig(SigMonitor(), actor)


# Kill an actor

proc kill*(actor: Actor) =
  actor.sendSig(SigKill(), Actor())


# Signal termination of an actor; inform the parent and kill any linked
# actors.

proc exit(pool: ptr Pool, actor: Actor, reason: ExitReason, ex: ref Exception = nil) =

  actor.withLock:
    if actor[].state == Dead:
      return
    if reason == Lost:
      echo actor, " was lost in state ", actor[].state
    actor[].state = Dead

  actor.handleSignals()

  # Inform the parent of the death of their child
  if actor[].monitors.len > 0:
    # getCurrentException() returns a ref to a global .threadvar, which is not
    # safe to carry around. As a workaround, create a fresh exception and copy
    # some of the fields. TODO: what about the stack frames?
    for monitor in actor[].monitors:
      var ex: ref Exception
      let exCur = getCurrentException()
      if not exCur.isNil:
        ex = new Exception
        ex.name = exCur.name
        ex.msg = exCur.msg
      monitor.sendSig(MessageExit(actor: actor, reason: reason, ex: move ex), Actor())
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

  actor.withLock:
    actor[].sigQueue.clear()
    # Drop the continuation
    if not actor[].c.isNil:
      disarm actor[].c
      reset actor[].c

  # These memberse are not usually locked, need to be locked at cleanup time to
  # make sure the call to exit() from `=destroy` is synchronized.
  actor.withLock:
    actor[].msgQueue.clear()
    actor[].links.setLen(0)
    actor[].monitors.setLen(0)

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
    pool.withLock:
      while pool.workQueue.len == 0 and not pool.stop:
        pool.cond.wait(pool.lock)
      if pool.stop:
        break
      actor = pool.workQueue.popFirst()

    # Update actor state to 'Running' and take out the continuation
    # so it can be trampolined
    actor.withLock:
      if actor[].state == Dead:
        continue
      doAssert actor[].state == Queued
      actor[].state = Running
      c = move actor[].c

    # Check the actor's signal queue
    actor.handleSignals()

    block crashed:
      # Trampoline the actor's continuation
      try:
        {.cast(gcsafe).}:
          while not c.isNil and not c.fn.isNil:
            let fn = c.fn
            var c2 = fn(c)
            c = move c2
      except:
        {.emit: ["/* actor crashed */"].}
        pool.exit(actor, Error)
        break crashed

      # Handle new actor state
      var state: State
      actor.withLock:
        state = actor[].state
      if c.finished:
        {.emit: ["/* actor finished */"].}
        pool.exit(actor, Normal)
      elif state == Suspending:
        pool.withLock:
          actor.withLock:
            # if there are already new signals in the queue, the actor needs
            # to be queued again right away
            if actor[].sigQueue.len > 0:
              actor[].state = Queued
              pool.workQueue.addLast(actor)
              pool.cond.signal()
            else:
              actor[].state = Suspended
      elif state == Jielding:
        pool.withLock:
          actor.withLock:
            actor[].state = Queued
            pool.workQueue.addLast(actor)
            pool.cond.signal()
      elif state == Killed:
        {.emit: ["/* actor killed */"].}
        pool.exit(move actor, Killed)


proc hatchAux*(pool: ptr Pool, c: sink ActorCont, parent=Actor(), linked=false): Actor =

  assert not isNil(c)
  #assertIsolated(c)

  let a = create(ActorObject)
  a.pid = pool.actorPidCounter.fetchAdd(1)
  a.pool = pool
  a.rc.store(0)
  a.lock.initLock()
  var actor = Actor(p: a)

  c.pool = pool
  c.actor = actor
  actor.withLock:
    actor[].c = move c

  if not parent.isNil:
    a.monitors.add(parent)

  if linked:
    link(actor, parent)

  pool.actorCount += 1
  pool.resume(actor)

  actor


# Create pool with actor queue and worker threads

proc newPool*(nWorkers: int = countProcessors()): ref Pool =

  var pool = new Pool
  initLock pool.lock
  pool.withLock:
    initCond pool.cond
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

  pool.withLock:
    pool.stop = true
    pool.cond.broadcast()

  for worker in pool.workers:
    worker.thread.joinThread()
    assertIsolated(worker)



