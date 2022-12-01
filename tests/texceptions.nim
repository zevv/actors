
# This used to trigger drd/helgrind when pool used to pass the getCurrentException() result,
# which is a ref to a global threadvar inside the nim stdlib
#
# nim c --verbosity:0 -d:usemalloc --mm:arc --debugger:native -d:danger tests/texceptions
# valgrind --error-exitcode=255 --quiet --tool=drd --suppressions=./misc/valgrind-drd-suppressions tests/texceptions
#
# ==58541== Possible data race during write of size 8 at 0x4A8A0E0 by thread #6
# ==58541== Locks held: none
# ==58541==    at 0x10C35C: nimDecRefIsLast (arc.nim:183)
# ==58541==    by 0x10C35C: eqdestroy___stdZassertions_27 (assertions.nim:31)
# ==58541==    by 0x10ECAA: workerThread__OOZnimactorsZpool_1327 (assertions.nim:31)
# ==58541==    by 0x10BD3D: threadProcWrapDispatch__OOZnimactorsZpool_1860 (threadimpl.nim:74)
# ==58541==    by 0x109F70: threadProcWrapper__OOZnimactorsZpool_1836 (threadimpl.nim:106)
# ==58541==    by 0x484E7D6: mythread_wrapper (hg_intercepts.c:406)
# ==58541==    by 0x490A849: start_thread (pthread_create.c:442)
# ==58541==    by 0x498D52F: clone (clone.S:100)
# ==58541== 
# ==58541== This conflicts with a previous write of size 8 by thread #8
# ==58541== Locks held: none
# ==58541==    at 0x10D7F1: nimDecRefIsLast (arc.nim:183)
# ==58541==    by 0x10D7F1: eqdestroy___OOZnimactorsZpool_1296 (pool.nim:252)
# ==58541==    by 0x10A2F0: nimDestroyAndDispose (arc.nim:152)
# ==58541==    by 0x10A2F0: nimDestroyAndDispose (arc.nim:152)
# ==58541==    by 0x10ED0B: workerThread__OOZnimactorsZpool_1327 (transform.nim:1061)
# ==58541==    by 0x10BD3D: threadProcWrapDispatch__OOZnimactorsZpool_1860 (threadimpl.nim:74)
# ==58541==    by 0x109F70: threadProcWrapper__OOZnimactorsZpool_1836 (threadimpl.nim:106)
# ==58541==    by 0x484E7D6: mythread_wrapper (hg_intercepts.c:406)
# ==58541==    by 0x490A849: start_thread (pthread_create.c:442)


import std/os

import nimactors

proc claire() {.actor.} =
  raise newException(IOError, "flap")

proc main() {.actor.} =
  let pid = hatch claire()
  let m = recv(MessageExit)
  doAssert m.reason == Error


for i in 1..100:
  var pool = newPool(4)
  discard pool.hatch main()
  pool.join()

