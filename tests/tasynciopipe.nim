
when defined(cpp):
  echo "https://github.com/nim-lang/Nim/issues/20081"
  quit 0

# Send data between pairs of actors through a unix pipe 

import std/os
import std/syncio
import std/strformat
import std/strutils
import std/times
import std/posix
import std/atomics

proc pipe2*(a: array[0..1, cint], flags: cint): cint {.importc, header: "<unistd.h>".}

import pkg/nimactors/lib/asyncio

var rtotal: Atomic[int]
var wtotal: Atomic[int]

const chunkSize = 1024 * 1024
const chunkCount = 1024

proc reader(evq: Evq, fd: cint, n: int) {.actor.} =
  var i: int
  var buf = newString(chunkSize)
  while i < n:
    let r = evq.readAll(fd, buf[0].addr, buf.len)
    rtotal += r
    if i mod 128 == 0:
      echo "rtotal ", rtotal
    inc i


proc writer(evq: Evq, fd: cint, n: int) {.actor.} =
  var i: int
  var buf = newString(chunkSize)
  while i < n:
    let r = evq.writeAll(fd, buf[0].addr, buf.len)
    wtotal += r
    inc i


proc main() {.actor.} =
 
  let evq = newEvq()
  let pipes = 16
  var i = 0

  while i < pipes:
    var fds: array[2, cint]
    discard pipe2(fds, O_NONBLOCK)
    discard hatch reader(evq, fds[0], chunkCount)
    discard hatch writer(evq, fds[1], chunkCount)
    inc i

  echo "waiting for actors to finish"
  while i > 0:
    discard recv(MessageExit)
    discard recv(MessageExit)
    dec i

  evq.stop()

  doAssert rtotal.load() == wtotal.load()
  doAssert rtotal.load() == pipes * chunkCount * chunkSize
  echo rtotal.load() / (1024 * 1024 * 1024), " Gb tranferred"


proc go() =
  var pool = newPool(16)
  discard pool.hatch main()
  pool.join()

go()


