
import std/epoll
import std/math
import std/posix
import std/tables
import std/locks
import std/heapqueue

import actors
import actors/lib/evq

export actors
export evq

proc sleep*(evq: Evq, interval: float) {.actor.} =
  evq.addTimer(interval)
  discard recv(MessageEvqTimer)


proc read*(evq: Evq, fd: cint, buf: ptr char, size: int): int {.actor.} =
  evq.addFd(fd, POLLIN)
  let p = cast[ptr char](cast[Byteaddress](buf))
  result = posix.read(fd, p, size)
  evq.delFd(fd)


proc write*(evq: Evq, fd: cint, buf: ptr char, size: int): int {.actor.} =
  evq.addFd(fd, POLLIN)
  let p = cast[ptr char](cast[Byteaddress](buf))
  result = posix.write(fd, p, size)
  evq.delFd(fd)


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
      if r < size-done:
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
      if r < size-done:
        break
  evq.delFd(fd)
  result = done

