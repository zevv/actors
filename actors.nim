
import std/tables
import std/rlocks

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
    lock: RLock
    actors: Table[string, Actor]

  ActorProc = proc(a: Actor) {.thread, nimcall.}


var pool: ActorPool
initRLock pool.lock


proc spawn*(name: string, fn: ActorProc) =
  let a = Actor(name: name)
  a.mailbox.open()
  createThread(a.thread, fn, a)
  withRLock pool.lock:
    pool.actors[name] = a


# Look up the actor in the global `actors` table. `actors` is owned by the main
# thread so we need a gcsafe cast and a raw pointer to be able to be able to
# use it and make sure the calling thread does not touch the RC.

proc findActor(name: string): Actor =
  {.cast(gcsafe).}:
    withRLock pool.lock:
      if name in pool.actors:
        result = pool.actors[name]

template ifItIsActorNamed(name: string; body: untyped): untyped =
  {.cast(gcsafe).}:
    withRLock pool.lock:
      var it {.inject.} = findActor(name)
      if not it.isNil:
        body

proc send*(a: Actor, name: string, m: sink Message) =
  # Tell nim that the object moved away and it should not touch the RC
  # after this procedure
  defer:
    wasmoved(m)
  # Set the source name of the message
  m.src = a.name
  # It is only safe to send the message if ours is the only reference to it
  verifyIsolated(m)
  ifItIsActorNamed name:
    it.mailbox.send(m)
    return
  raise ValueError.newException name & " is dead"


proc recv*(a: Actor): Message =
  result = a.mailbox.recv()


proc joinAll*() =
  for _, a in pool.actors:
    a.thread.joinThread()

