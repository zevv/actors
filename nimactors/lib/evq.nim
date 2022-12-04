
import std/epoll
import std/posix
import std/monotimes
import std/tables
import std/heapqueue

import ../../nimactors

type

  Evq* = distinct Actor

  EvqImpl = ref object
    epfd: cint
    fds: Table[cint, Actor]
    pool: ptr Pool
    now: float
    timers*: HeapQueue[Timer]

  Timer = object
    time: float
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
    evq.fds[m.MessageEvqAddFd.fd] = m.src

  elif m of MessageEvqDelFd:
    let m = m.MessageEvqDelFd
    discard epoll_ctl(evq.epfd, EPOLL_CTL_DEL, m.fd.cint, nil)
    evq.fds.del(m.fd)

  else:
    discard
    #echo "unhandled message"


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

  var evq = EvqImpl(epfd: epoll_create(1))

  #
  var ee = EpollEvent(events: POLLIN.uint32, data: EpollData(u64: fdWake.uint64))
  discard epoll_ctl(evq.epfd, EPOLL_CTL_ADD, fdWake, ee.addr)

  while true:

    var es: array[8, EpollEvent]
    let timeout = evq.calculateTimeout()
    #echo "epollin"
    let n = epoll_wait(evq.epfd, es[0].addr, es.len.cint, timeout)
    #echo "epollout ", n

    evq.now = getMonoTime().ticks.float / 1.0e9
    evq.handleTimers()

    var i = 0
    while i < n:
      let fd = es[i].data.u64.cint

      if fd == fdWake:
        var b: array[1024, char]
        var n = posix.read(fdWake, b.addr, sizeof(b))

      elif fd in evq.fds:
        let actor = evq.fds[fd]
        send(actor, MessageEvqEvent())

      inc i

    # Handle messages

    while true:
      let m = tryRecv()
      if not m.isNil:
        evq.handleMessage(m)
      else:
        break

    jield()


# Public API

proc addTimer*(c: ActorCont, evq: Evq, interval: float) {.cpsVoodoo.} =
  evq.Actor.send(MessageEvqAddTimer(interval: interval), c.actor)


proc addFd*(c: ActorCont, evq: Evq, fd: cint, events: cshort) {.cpsVoodoo.} =
  evq.Actor.send(MessageEvqAddFd(fd: fd, events: events), c.actor)


proc delFd*(c: ActorCont, evq: Evq, fd: cint) {.cpsVoodoo.} =
  evq.Actor.send(MessageEvqDelFd(fd: fd), c.actor)


proc sleep*(evq: Evq, interval: float) {.actor.} =
  evq.addTimer(interval)
  discard recv(MessageEvqEvent)


proc readAll*(evq: Evq, fd: cint, buf: ptr char, size: int): int {.actor.} =
  var done: int
  evq.addFd(fd, POLLIN)
  while done < size:
    discard recv(MessageEvqEvent)
    while done < size:
      let p = cast[ptr char](cast[Byteaddress](buf) + done)
      let r = posix.read(fd, p, size-done)
      if r > 0:
        done += r
      elif r < size-done:
        break
  evq.delFd(fd)
  return done


proc writeAll*(evq: Evq, fd: cint, buf: ptr char, size: int): int {.actor.} =
  var done = 0
  evq.addFd(fd, POLLOUT)
  while done < size:
    discard recv(MessageEvqEvent)
    while done < size:
      let p = cast[ptr char](cast[Byteaddress](buf) + done)
      let r = posix.write(fd, p, size-done)
      if r > 0:
        done += r
      elif r < size-done:
        break
  evq.delFd(fd)
  result = done


proc kill*(actor: Evq) {.borrow.}


proc link*(actor: Actor, evq: Evq) =
  link(actor, evq.Actor)


proc newEvq*(): Evq {.actor.} =

  var fds: array[2, cint]
  discard pipe(fds)
  discard fcntl(fds[1], F_SETFL, O_NONBLOCK)

  var actor = hatch evqActor(fds[0])

  setSignalFd(actor, fds[1])
  actor.Evq



