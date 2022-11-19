
import tables
import std/locks
import os
import isisolated


type

  # orc can not handle data that has been moved over threads, so mark
  # the message as acyclic to keep orcs hands off it
  Message* {.acyclic.} = ref object of RootObj

  Actor* = ref object
    thread: Thread[Actor]
    mailbox: Channel[Message]

  ActorPool = object
    lock: Lock
    actors: Table[string, Actor]

  ActorProc = proc(a: Actor) {.thread,nimcall.}


var pool: ActorPool


proc spawn*(name: string, fn: ActorProc) =
  let a = Actor()
  a.mailbox.open()
  createThread(a.thread, fn, a)
  withLock pool.lock:
    pool.actors[name] = a


proc send*(name: string, m: sink Message) =
  # It is only safe to send the message if ours is the only reference to it
  verifyIsolated(m)
  # Look up the actor in the global `actors` table. `actors` is owned by the
  # main thread so we need a gcsafe cast and cursor to be able to be able to
  # use it and not touch its RC
  {.cast(gcsafe).}:
    var dst {.cursor.}: Actor
    withLock pool.lock:
      if name in pool.actors:
        dst = pool.actors[name]
        dst.mailbox.send(m)
        # Tell nim that the object moved away and it should not touch the RC
        # after this anymore
        wasmoved(m)


proc recv*(a: Actor): Message =
  echo "recv"
  result = a.mailbox.recv()


proc joinAll*() =
  for _, a in pool.actors:
    a.thread.joinThread()

