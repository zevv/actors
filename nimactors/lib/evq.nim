
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

  Evq* = ref object
    id*: ActorId

  EvqImpl = ref object
    fdWake: cint
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
  


proc addFd(evq: EvqImpl, fd: cint) =
  var ee = EpollEvent(events: POLLIN.uint32, data: EpollData(u64: fd.uint64))
  discard epoll_ctl(evq.epfd, EPOLL_CTL_ADD, fd, ee.addr)


proc handleMessage(evq: EvqImpl, m: Message) {.actor.} =

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
    evq.addFd m.MessageEvqAddFd.fd
    let io = Io(kind: iokFd, actorId: m.src)
    evq.ios[m.MessageEvqAddFd.fd] = io

  elif m of MessageEvqDelFd:
    let fd = m.MessageEvqDelFd.fd
    discard epoll_ctl(evq.epfd, EPOLL_CTL_DEL, fd.cint, nil)
    evq.ios.del(fd)
  
  else:
    echo "unhandled message"


# This actor is special, as it has to wait on both the epoll and the regular
# mailbox. This is done by adding a pipe-to-self, which is written by the
# send() when a message is posted to the mailbox

proc evqActor*(evq: EvqImpl) {.actor.} =
  
  while true:
        
    var es: array[8, EpollEvent]
    let n = epoll_wait(evq.epfd, es[0].addr, es.len.cint, -1)
    
    var i = 0
    while i < n:
      let fd = es[i].data.u64.cint

      if fd == evq.fdWake:
        var b: char
        discard posix.read(evq.fdWake, b.addr, 1)
        evq.handleMessage recv()

      elif fd in evq.ios:
        let io = evq.ios[fd]
        if io.kind == iokTimer:
          var data: uint64
          discard posix.read(fd, data.addr, sizeof(data))
        send(io.actorId, MessageEvqEvent())

      inc i


# Public API

proc addTimer*(actor: Actor, evq: Evq, interval: float) {.cpsVoodoo.} =
  send(actor.pool, actor.id, evq.id, MessageEvqAddTimer(interval: interval))


proc addFd*(actor: Actor, evq: Evq, fd: cint) {.cpsVoodoo.} =
  send(actor.pool, actor.id, evq.id, MessageEvqAddFd(fd: fd))


proc delFd*(actor: Actor, evq: Evq, fd: cint) {.cpsVoodoo.} =
  send(actor.pool, actor.id, evq.id, MessageEvqDelFd(fd: fd))


proc newEvq*(): Evq {.actor.} =

  var fds: array[2, cint]
  discard pipe(fds)
  
  var evq = EvqImpl(
    epfd: epoll_create(1),
    fdWake: fds[0]
  )
  
  evq.addFd(evq.fdWake)

  let id = hatch evqActor(evq)
  setMailboxFd(id, fds[1])

  Evq(id: id)



