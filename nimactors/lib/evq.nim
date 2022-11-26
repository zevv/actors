
import std/epoll
import std/posix
import std/tables

import cps

import ../../nimactors


proc timerfd_create(clock_id: ClockId, flags: cint): cint
     {.cdecl, importc: "timerfd_create", header: "<sys/timerfd.h>".}

proc timerfd_settime(ufd: cint, flags: cint,
                      utmr: var Itimerspec, otmr: var Itimerspec): cint
     {.cdecl, importc: "timerfd_settime", header: "<sys/timerfd.h>".}


type

  EvqInfo = ref object
    actorId*: ActorId
    fdWake*: cint

  Evq* = ref object
    epfd: cint
    ios: Table[cint, Io]
    pool: ptr Pool

  IoKind = enum
    iokTimer, iokFd

  Io = ref object
    kind: IoKind
    actorId: ActorId

  MessageEvqAddTimer* = ref object of Message
    interval: float
  
  MessageEvqAddFd* = ref object of Message
    fd: cint
  
  MessageEvqDelFd* = ref object of Message
    fd: cint
  
  MessageEvqEvent* = ref object of Message
  

proc handleMessage(evq: Evq) {.actor.} =
  var m = recv()

  if m of MessageEvqAddTimer:
    let interval = m.MessageEvqAddTimer.interval
    let fd = timerfd_create(CLOCK_MONOTONIC, O_CLOEXEC or O_NONBLOCK).cint
    var newTs, oldTs: Itimerspec   
    newTs.it_interval.tv_sec = posix.Time(0)
    newTs.it_interval.tv_nsec = (interval * 1000 * 1000 * 1000).clong
    newTs.it_value.tv_sec = newTs.it_interval.tv_sec
    newTs.it_value.tv_nsec = newTs.it_interval.tv_nsec       
    discard timerfd_settime(fd, cint(0), newTs, oldTs)
    var ee2 = EpollEvent(events: POLLIN.uint32, data: EpollData(u64: fd.uint64))
    discard epoll_ctl(evq.epfd, EPOLL_CTL_ADD, fd.cint, ee2.addr)
    let io = Io(kind: iokTimer, actorId: m.src)
    evq.ios[fd] = io

  elif m of MessageEvqAddFd:
    let fd = m.MessageEvqAddFd.fd
    var ee2 = EpollEvent(events: POLLIN.uint32 or EPOLLET.uint32, data: EpollData(u64: fd.uint64))
    discard epoll_ctl(evq.epfd, EPOLL_CTL_ADD, fd.cint, ee2.addr)
    let io = Io(kind: iokFd, actorId: m.src)
    evq.ios[fd] = io

  elif m of MessageEvqDelFd:
    let fd = m.MessageEvqDelFd.fd
    discard epoll_ctl(evq.epfd, EPOLL_CTL_DEL, fd.cint, nil)
    evq.ios.del(fd)
  
  else:
    echo "unhandled message"


# This actor is special, as it has to wait on both the epoll and the regular
# mailbox. This is done by adding a pipe-to-self, which is written by the
# send() when a message is posted to the mailbox

proc evqActor(fdWake: cint) {.actor.} =

  var evq = Evq(
    epfd: epoll_create(1)
  )

  var ee = EpollEvent(events: POLLIN.uint32, data: EpollData(u64: fdWake.uint64))
  discard epoll_ctl(evq.epfd, EPOLL_CTL_ADD, fdWake.cint, ee.addr)

  while true:

    var es: array[8, EpollEvent]
    let n = epoll_wait(evq.epfd, es[0].addr, es.len.cint, 1000)

    var i = 0
    while i < n:
      let fd = es[i].data.u64.cint

      if fd == fdWake:
        var b: char
        discard posix.read(fdWake, b.addr, 1)
        handleMessage(evq)

      elif fd in evq.ios:
        let io = evq.ios[fd]
        if io.kind == iokTimer:
          var data: uint64
          discard posix.read(fd, data.addr, sizeof(data))
        send(io.actorId, MessageEvqEvent())

      inc i


proc addTimer*(actor: Actor, interval: float) {.cpsVoodoo.} =
  let msg = MessageEvqAddTimer(interval: interval)
  send(actor.pool, actor.id, actor.pool.evqActorId, msg)


proc addFd*(actor: Actor, fd: cint) {.cpsVoodoo.} =
  let msg = MessageEvqAddFd(fd: fd)
  send(actor.pool, actor.id, actor.pool.evqActorId, msg)


proc delFd*(actor: Actor, fd: cint) {.cpsVoodoo.} =
  let msg = MessageEvqDelFd(fd: fd)
  send(actor.pool, actor.id, actor.pool.evqActorId, msg)


proc newEvq*(pool: ref Pool): EvqInfo =

  var fds: array[2, cint]
  discard pipe(fds)

  new result
  result.actorId = pool.hatch evqActor(fds[0])
  result.fdWake = fds[1]

