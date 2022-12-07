
import std/epoll
import std/math
import std/posix
import std/tables
import std/locks
import std/heapqueue

import nimactors

type

  Evq* = distinct Actor

  EvqImpl = ref object
    actor: Actor
    workers: seq[ref EvqWorker]
    timerFd: cint
    timers*: HeapQueue[Timer]
    fds: Table[cint, Actor]

  EvqWorker = object
    id: int
    actor: Actor
    epfd: cint
    thread: Thread[ptr EvqWorker]

  Timer = object
    t_when: float
    actor: Actor

  MessagePollResult = ref object of Message
    count: cint
    fds: array[32, cint]

  MessageEvqAddTimer* = ref object of Message
    interval: float

  MessageEvqAddFd* = ref object of Message
    fd: cint
    events: cshort

  MessageEvqDelFd* = ref object of Message
    fd: cint

  MessageEvqEvent* = ref object of Message
    fd: cint

  MessageEvqTimer* = ref object of Message


template `<`(a, b: Timer): bool =
  a.t_when < b.t_when



proc timerfd_create(clock_id: ClockId, flags: cint): cint
     {.cdecl, importc: "timerfd_create", header: "<sys/timerfd.h>".}

proc timerfd_settime(ufd: cint, flags: cint,                      
                      utmr: var Itimerspec, otmr: var Itimerspec): cint
     {.cdecl, importc: "timerfd_settime", header: "<sys/timerfd.h>".}

const TFD_TIMER_ABSTIME: cint = 1


# The poll thread is pretty simple: it waits on the pollfd, and sends the
# results of epoll_wait() to the actor, where the individual fds and timers are
# handled.

proc workerThread(ew: ptr EvqWorker) =

  while true:

    var es: array[32, EpollEvent]
    let n = epoll_wait(ew.epfd, es[0].addr, es.len.cint, -1)

    if n > 0:
      let m = MessagePollResult(count: n)
      var i = 0
      while i < n:
        m.fds[i] = es[i].data.u64.cint
        inc i
      sendSig(ew.actor, m, ew.actor)

    else:
      if errno != EINTR:
        echo "epoll_wait error"
        quit 1



proc getMonotime(): float =
  var ts: Timespec
  discard clock_gettime(CLOCK_MONOTONIC, ts)
  result = ts.tv_sec.float + ts.tv_nsec.float / 1.0e9


# Update the timerfd to reflect the first expiring timer from the heap queue

proc updateTimer(ei: EvqImpl) =
    
  var oldTs, newTs: Itimerspec

  if ei.timers.len > 0:
    let t = ei.timers[0]
    newTs.it_value.tv_sec = posix.Time(t.t_when)
    newTs.it_value.tv_nsec = (math.mod(t.t_when, 1.0) * 1.0e9).clong

  let r = timerfd_settime(ei.timerfd, TFD_TIMER_ABSTIME, newTs, oldTs)
  if r != 0:
    echo "timerfd_settime error"
    quit 1


# Handle and remove all expired timers from the heap queue

proc handleTimers(ei: EvqImpl) =
  let t_now = getMonoTime()
  while ei.timers.len > 0 and t_now >= ei.timers[0].t_when:
    let t = ei.timers.pop()
    sendSig(t.actor, MessageEvqTimer(), ei.actor)


# Map the file descriptor to a worker ID. For now, this is flat
# and simple, might need tweaking in the future to avoid stupid
# distributions

proc mapFdToWorkerId(ei: EvqImpl, fd: cint): int =
  fd mod ei.workers.len



# Event queue actor implementation

proc evqActor*(nWorkers: int) {.actor.} =

  # Create event queue instance

  var ei = EvqImpl(
    actor: self(),
    timerFd: timerfd_create(CLOCK_MONOTONIC, O_NONBLOCK),
  )

  # Create worker threads

  var i = 0
  while i < nWorkers:

    var ew = new EvqWorker
    ew.id = i
    ew.actor = self()
    ew.epfd = epoll_create(1)

    var ee = EpollEvent(events: POLLIN.uint32, data: EpollData(u64: ei.timerfd.uint64))
    discard epoll_ctl(ew.epfd, EPOLL_CTL_ADD, ei.timerfd, ee.addr)

    createThread(ew.thread, workerThread, ew[].addr)

    ei.workers.add ew
    inc i

  # Actor main loop handles requests from other actors

  receive:

    (fds, n) = MessagePollResult(count: n, fds: fds):
      var i = 0
      while i < n:
        let fd = fds[i]
        if fd == ei.timerFd:
          var val: uint64
          discard posix.read(ei.timerFd, val.addr, sizeof(val))
          ei.handleTimers()
        else:
          if fd in ei.fds:
            let actor = ei.fds[fd]
            send(actor, MessageEvqEvent())
        inc i

    (interval, src) = MessageEvqAddTimer(interval: interval, src: src):
      let t_now = getMonoTime()
      ei.timers.push Timer(actor: src, t_when: t_now + interval)
      ei.updateTimer()

    (fd, events, src) = MessageEvqAddFd(fd: fd, events: events, src: src):
      var ee = EpollEvent(events: events.uint32 or EPOLLET.uint32, data: EpollData(u64: fd.uint64))
      let wid = ei.mapFdToWorkerId(fd)
      let r = epoll_ctl(ei.workers[wid].epfd, EPOLL_CTL_ADD, fd, ee.addr)
      if r == 0:
        ei.fds[fd] = src
      else:
        echo "epoll_ctl error ", strerror(errno)
        quit 1

    (fd, src) = MessageEvqDelFd(fd: fd, src: src):
      let wid = ei.mapFdToWorkerId(fd)
      discard epoll_ctl(ei.workers[wid].epfd, EPOLL_CTL_DEL, fd, nil)
      ei.fds.del(fd)



# Public API

proc addTimer*(evq: Evq, interval: float) {.actor.} =
  send(evq.Actor, MessageEvqAddTimer(interval: interval))


proc addFd*(evq: Evq, fd: cint, events: cshort) {.actor.} =
  send(evq.Actor, MessageEvqAddFd(fd: fd, events: events))


proc delFd*(evq: Evq, fd: cint) {.actor.} =
  send(evq.Actor, MessageEvqDelFd(fd: fd))


proc kill*(actor: Evq) {.borrow.}


proc link*(actor: Actor, evq: Evq) =
  link(actor, evq.Actor)


proc newEvq*(): Evq {.actor.} =
  var actor = hatch evqActor(2)
  actor.Evq



