
# Send 1 GByte of data between two actors through a unix pipe 

import std/os
import std/syncio
import std/strformat
import std/strutils
import std/times
import std/posix
import std/atomics

proc pipe2*(a: array[0..1, cint], flags: cint): cint {.importc, header: "<unistd.h>".}


import nimactors
import nimactors/lib/evq
  
const bytes = 1024 * 1024 * 1024


proc reader(evq: Evq, fd: cint, bytes: int) {.actor.} =
  var bytesReceived = 0
  while bytesReceived < bytes:
    var buf = newString(1024 * 1024)
    let r = evq.read(fd, buf[0].addr, buf.len)
    if r > 0:
      bytesReceived += r



proc writer(evq: Evq, fd: cint, bytes: int) {.actor.} =
  let blob = newString(1024 * 1024)
  var loops = bytes /% blob.len
  while loops > 0:
    let r = evq.write(fd, blob[0].addr, blob.len)
    dec loops


proc main() {.actor.} =
 
  let evq = newEvq()
  let pipes = 8
  var i = 0

  while i < pipes:
    var fds: array[2, cint]
    #discard pipe2(fds, O_NONBLOCK)
    discard posix.socketpair(AF_UNIX, SOCK_STREAM or O_NONBLOCK, 0, fds)

    let r = hatch reader(evq, fds[0], bytes)
    let w = hatch writer(evq, fds[1], bytes)
    inc i

  echo "waiting for actors to finish"
  i = 0
  while i < pipes:
    discard recv(MessageExit)
    discard recv(MessageExit)
    inc i

  echo "killing evq"
  kill evq


proc go() =
  var pool = newPool(16)
  discard pool.hatch main()
  pool.join()

go()


