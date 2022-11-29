
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
    actor: Actor

  IoKind = enum
    iokTimer, iokFd

  Io = ref object
    kind: IoKind
    actor: Actor

  MessageEvqAddTimer* = ref object of Message
    interval: float
  
  MessageEvqAddFd* = ref object of Message
    fd: cint
    events: cshort
  
  MessageEvqDelFd* = ref object of Message
    fd: cint
  
  MessageEvqEvent* = ref object of Message
  

template `<`(a, b: Timer): bool =
  a.time < b.time


proc handleMessage(evq: EvqImpl, m: Message) {.actor.} =

  if m of MessageEvqAddTimer:
    let m = m.MessageEvqAddTimer
    evq.timers.push Timer(actor: m.src, time: evq.now + m.interval)

  elif m of MessageEvqAddFd:
    let m = m.MessageEvqAddFd
    var ee = EpollEvent(events: m.events.uint32 or EPOLLET.uint32, data: EpollData(u64: m.fd.uint64))
    discard epoll_ctl(evq.epfd, EPOLL_CTL_ADD, m.fd, ee.addr)
    let io = Io(kind: iokFd, actor: m.src)
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


template handleTimers(evq: EvqImpl) =
  evq.updateNow()
  while evq.timers.len > 0 and evq.now >= evq.timers[0].time:
    let t = evq.timers.pop
    send(t.actor, MessageEvqEvent())


# This actor is special, as it has to wait on both the epoll and the regular
# mailbox. This is done by adding a pipe-to-self, which is written by the
# send() when a message is posted to the mailbox

proc evqActor*(fdWake: cint) {.actor.} =
  
  var evq = EvqImpl(
    epfd: epoll_create(1),
  )
  
  var ee = EpollEvent(events: POLLIN.uint32, data: EpollData(u64: fdWake.uint64))
  discard epoll_ctl(evq.epfd, EPOLL_CTL_ADD, fdWake, ee.addr)
  
  while true:
        
    var es: array[8, EpollEvent]
    let timeout = evq.calculateTimeout()
    let n = epoll_wait(evq.epfd, es[0].addr, es.len.cint, timeout)

    evq.now = getMonoTime().ticks.float / 1.0e9
    evq.handleTimers()
    
    var i = 0
    while i < n:
      let fd = es[i].data.u64.cint

      if fd == fdWake:
        var b: array[1024, char]
        var n = posix.read(fdWake, b.addr, sizeof(b))

      elif fd in evq.ios:
        let io = evq.ios[fd]
        if io.kind == iokTimer:
          var data: uint64
          discard posix.read(fd, data.addr, sizeof(data))
        send(io.actor, MessageEvqEvent())

      inc i

    # Handle messages
    
    while true:
      let m = tryRecv()
      if not m.isNil:
        evq.handleMessage(m)
      else:
        break

# Public API


proc addTimer*(c: ActorCont, evq: Actor, interval: float) {.cpsVoodoo.} =
  send(c.pool, c.actor, evq, MessageEvqAddTimer(interval: interval))


proc addFd*(c: ActorCont, evq: Actor, fd: cint, events: cshort) {.cpsVoodoo.} =
  send(c.pool, c.actor, evq, MessageEvqAddFd(fd: fd, events: events))


proc delFd*(c: ActorCont, evq: Actor, fd: cint) {.cpsVoodoo.} =
  send(c.pool, c.actor, evq, MessageEvqDelFd(fd: fd))


proc sleep*(evq: Actor, interval: float) {.actor.} =
#template sleep*(evq: Actor, interval: float) =
  evq.addTimer(interval)
  discard recv(MessageEvqEvent)


proc read*(evq: Actor, fd: cint, buf: ptr char, size: int): int {.actor.} =
  evq.addFd(fd, POLLIN)
  discard recv(MessageEvqEvent)
  result = posix.read(fd, buf, size)
  evq.delFd(fd)


proc newEvq*(): Actor {.actor.} =

  var fds: array[2, cint]
  discard pipe(fds)
  discard fcntl(fds[1], F_SETFL, O_NONBLOCK)
 

  var actor = hatch evqActor(fds[0])

  #defer:
  #  reset actor

  setSignalFd(actor, fds[1])
  actor
  


