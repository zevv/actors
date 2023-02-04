
#
# Minimal example showing moving of data between two messages
# using a simple cond+mutex+deque queue
#

import std/locks
import std/deques
import std/os

const count = 100000

type
  Message = ref object
    val: int

  Queue = object
    lock: Lock
    cond {.guard:lock.}: Cond
    messages {.guard:lock.}: Deque[Message]


proc push(queue: ptr Queue, msg: sink Message) =
  withLock queue.lock:
    queue.messages.addLast(msg)
    queue.cond.signal()

proc pop(queue: ptr Queue): Message =
  withLock queue.lock:
    while queue.messages.len == 0:
      queue.cond.wait(queue.lock)
    result = queue.messages.popFirst()


proc foo(queue: ptr Queue) {.thread.} =
  for i in 1..count:
    let m1 = queue.pop()
    echo getThreadId(), " ", m1.val

    let m2 = new Message
    m2.val = m1.val + 1
    queue.push(m2)

    os.sleep(1)



proc main() =
    

  var queue: Queue
  initLock queue.lock
  withLock queue.lock:
    initCond queue.cond
  
  var thread1: Thread[ptr Queue]
  var thread2: Thread[ptr Queue]

  createThread(thread1, foo, queue.addr)
  createThread(thread2, foo, queue.addr)
    
  queue.addr.push(Message())

  thread1.joinThread
  thread2.joinThread


main()
