
import tables
import std/locks
import os
import isisolated


type

  # orc can not handle data that has been moved over threads, so mark
  # the message as acyclic to keep orcs hands off it
  Message* {.acyclic.} = ref object of RootObj
    src*: string

  Actor* = ref object
    name: string
    thread: Thread[Actor]
    mailbox: Channel[Message]

  ActorPool = object
    lock: Lock
    actors: Table[string, Actor]

  ActorProc = proc(a: Actor) {.thread,nimcall.}


var pool: ActorPool


proc spawn*(name: string, fn: ActorProc) =
  let a = Actor(name: name)
  a.mailbox.open()
  createThread(a.thread, fn, a)
  withLock pool.lock:
    pool.actors[name] = a


proc send*(a: Actor, name: string, m: sink Message) =
  # Set the source name of the message
  m.src = a.name
  # It is only safe to send the message if ours is the only reference to it
  verifyIsolated(m)
  # Look up the actor in the global `actors` table. `actors` is owned by the
  # main thread so we need a gcsafe cast and cursor to be able to be able to
  # use it and make sure the calling thread does not touch the RC.
  {.cast(gcsafe).}:
    var dst {.cursor.}: Actor
    withLock pool.lock:
      if name in pool.actors:
        dst = pool.actors[name]
    if dst != nil:
      dst.mailbox.send(m)
      # Tell nim that the object moved away and it should not touch the RC
      # after this anymore
      wasmoved(m)


proc recv*(a: Actor): Message =
  result = a.mailbox.recv()


proc joinAll*() =
  for _, a in pool.actors:
    a.thread.joinThread()

