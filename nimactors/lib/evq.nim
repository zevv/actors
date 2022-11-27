
import std/epoll
import std/posix
import std/monotimes
import std/tables
import std/heapqueue

import ../../nimactors

type

  EvqImpl = ref object
    epfd: cint
    ios: Table[cint, Io]
    pool: ptr Pool
    now: float
    timers*: HeapQueue[Timer]

  Timer = object
    time: float
    actorId: Actor

  IoKind = enum
    iokTimer, iokFd

  Io = ref object
    kind: IoKind
    actorId: Actor

  MessageEvqAddTimer* = ref object of Message
    interval: float
  
  MessageEvqAddFd* = ref object of Message
    fd: cint
  
  MessageEvqDelFd* = ref object of Message
    fd: cint
  
  MessageEvqEvent* = ref object of Message
  

template `<`(a, b: Timer): bool =
  a.time < b.time


proc addFd(evq: EvqImpl, fd: cint) =
  var ee = EpollEvent(events: POLLIN.uint32 or EPOLLET.uint32, data: EpollData(u64: fd.uint64))
  discard epoll_ctl(evq.epfd, EPOLL_CTL_ADD, fd, ee.addr)


proc handleMessage(evq: EvqImpl, m: Message) {.actor.} =

  if m of MessageEvqAddTimer:
    let interval = m.MessageEvqAddTimer.interval
    evq.timers.push Timer(actorId: m.src, time: evq.now + interval)

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

   
proc updateNow(evq: EvqImpl) =
    evq.now = getMonoTime().ticks.float / 1.0e9


proc calculateTimeout(evq: EvqImpl): cint =
  evq.updateNow()
  result = -1
  if evq.timers.len > 0:
    let timer = evq.timers[0]
    result = cint(1000 * (timer.time - evq.now + 0.005))
    result = max(result, 0)


proc handleTimers(evq: EvqImpl) {.actor.} =
  evq.updateNow()
  while evq.timers.len > 0 and evq.timers[0].time <= evq.now:
    let t = evq.timers.pop
    #echo "Expired, send ", t.actorId
    send(t.actorId, MessageEvqEvent())


# This actor is special, as it has to wait on both the epoll and the regular
# mailbox. This is done by adding a pipe-to-self, which is written by the
# send() when a message is posted to the mailbox

proc evqActor*(fdWake: cint) {.actor.} =
  
  var evq = EvqImpl(
    epfd: epoll_create(1),
  )
  
  evq.addFd(fdWake)
  
  while true:
        
    var es: array[8, EpollEvent]
    let timeout = evq.calculateTimeout()
    #echo "timeout ", timeout
    let n = epoll_wait(evq.epfd, es[0].addr, es.len.cint, timeout)

    evq.now = getMonoTime().ticks.float / 1.0e9
    evq.handleTimers()
    
    var i = 0
    while i < n:
      let fd = es[i].data.u64.cint

      if fd == fdWake:
        var b: char
        discard posix.read(fdWake, b.addr, 1)
        evq.handleMessage recv()

      elif fd in evq.ios:
        let io = evq.ios[fd]
        if io.kind == iokTimer:
          var data: uint64
          discard posix.read(fd, data.addr, sizeof(data))
        send(io.actorId, MessageEvqEvent())

      inc i


# Public API

type

  Evq* = ref object
    id*: Actor


proc addTimer*(c: ActorCond, evq: Evq, interval: float) {.cpsVoodoo.} =
  send(c.pool, c.actor, evq.id, MessageEvqAddTimer(interval: interval))


proc addFd*(c: ActorCond, evq: Evq, fd: cint) {.cpsVoodoo.} =
  send(c.pool, c.actor, evq.id, MessageEvqAddFd(fd: fd))


proc delFd*(c: ActorCond, evq: Evq, fd: cint) {.cpsVoodoo.} =
  send(c.pool, c.actor, evq.id, MessageEvqDelFd(fd: fd))


proc sleep*(evq: Evq, interval: float) {.actor.} =
  evq.addTimer(interval)
  discard recv(MessageEvqEvent)


proc newEvq*(): Evq {.actor.} =

  var fds: array[2, cint]
  discard pipe(fds)
  
  let id = hatch evqActor(fds[0])
  setMailboxFd(id, fds[1])

  Evq(id: id)



