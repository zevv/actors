
This is an experimental project to create a threaded, share-nothing actor based
framework on top of CPS

This is all very much at an alpha stage; APIs are not likely to stay stable,
names of stuff might change, etc.

## Limitations

At this time, actors do not work with Nim's `orc` memory management; be sure
to compile with `--mm:arc`.

## Introduction

The Actors library provides a platform for building applications with massive
concurrency, without burdening the users with the pain and suffering of "old
school" threading primitives like locks, wait conditions, semaphores, etc.

Actors are built on two basic primitives:

- "Processes": an actor process provides a separate flow of execution. The term
  "process" is not to be confused with an operating-system process; instead,
  Processes are lightweight and have a small memory footprint (hundreds of
  bytes), are fast to create and terminate, and the scheduling overhead is
  slow. Processes are closely related to goroutines in the Go programming
  language, or processes in Erlang/Elixir.

- "Messages": Every process comes with a mailbox which can be used by other
  processes to send messages. Messages are Nim objects that can contain any
  kind of data, and can be sent from one process to another. The current
  implementation sends messages by moving the data between threads instead of
  copying, and is thus fast and low overhead.


## Quickstart

This section explains the bare necessities to get started with actors. It will
briefly explain the parts and concepts of the library, and show simple examples
on how to use them.


### The pool, and your first process

The actors library has a small runtime that manages the processes and messages,
which is called a "pool"; The pool spawns a number of OS threads to do the
work, and manages and schedules the actor processes. The `hatch()` function can
now be used to create new processes; (`hatch` is analogue to the `go` statement
in Golang or `spawn` in Elixir)

```
proc hello() {.actor.} =
  echo "hello"

let pool = newPool()
let pid = pool.hatch hello()
pool.join()

```

In the above example a new process is spawned ("hatched"), with `proc hello()`
as its entry point. Note that the proc is marked with the `{.actor.}` pragma.
The pool will run in the background until `join()` is called; this will wait
for all the actors to terminate and clean up all the resources used.




### Valgrind

To make the library aware of being run with helgrind, the program needs to be
built with `-d:usesValgrind`, otherwise false positives could be reported.


### More to come

Sure.






Actors communicate by sending and receiving messages; a message can be sent to
another process with the `send()` proc.

  
## References

References and further reading:

- https://en.wikipedia.org/wiki/Actor_model
- https://www.erlang.org/doc/reference_manual/processes.html
