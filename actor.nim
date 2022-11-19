
import tables
import std/locks
import isisolated


type

  # orc can not handle data that has been moved over threads, so mark
  # the message as acyclic to keep orcs hands off it
  Message* {.acyclic.} = ref object of RootObj
    src*: string

  ActorObject* = object
    name: string
    thread: Thread[Actor]
    mailbox: Channel[Message]

  Actor* = ref ActorObject

  ActorPtr* = ptr ActorObject

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

  
# Look up the actor in the global `actors` table. `actors` is owned by the main
# thread so we need a gcsafe cast and a raw pointer to be able to be able to
# use it and make sure the calling thread does not touch the RC.

proc findActor(name: string): ActorPtr =
  {.cast(gcsafe).}:
    var dst {.cursor.}: Actor
    withLock pool.lock:
      if name in pool.actors:
        result = pool.actors[name][].unsafeAddr


proc send*(a: Actor, name: string, m: sink Message) =
  # Set the source name of the message
  m.src = a.name
  # It is only safe to send the message if ours is the only reference to it
  verifyIsolated(m)
  var dst = findActor(name)
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

