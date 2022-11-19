
# Stolen from lib/system/arc.nim


when defined(gcOrc):                                          
  const          
    #rcIncrement = 0b10000 # so that lowest 4 bits are not touched
    #rcMask = 0b1111                                                                      
    rcShift = 4      # shift by rcShift to get the reference counter
                             
else:    
  const                                                
    #rcIncrement = 0b1000 # so that lowest 3 bits are not touched
    #rcMask = 0b111
    rcShift = 3      # shift by rcShift to get the reference counter

type

  NotIsolatedError* = object of CatchableError

  RefHeader = object
    rc: int
    when defined(gcOrc):
      rootIdx: int
    when defined(nimArcDebug) or defined(nimArcIds):
      refId: int

  Cell = ptr RefHeader


template head(p: pointer): Cell =
  cast[Cell](cast[int](p) -% sizeof(RefHeader))



# Anything not ref, seq or array is isolated

proc isIsolated*[T: not (ref or seq or array or object)](v: T): bool =
  #echo "- ", typeof(v), " ", v.repr
  true


# Iterate all elements of seqs and arrays

proc isIsolated*[T: seq or array](vs: T): bool =
  #echo "- ",  vs.repr
  for v in vs:
    if not isIsolated(v):
      return false
  true


# Iterate all fields of objects

proc isIsolated*[T: object and not ref](v: T): bool =
  #echo "- ", v.repr
  for k, v in fieldPairs(v):
    if not isIsolated(v):
      return false
  true


# Check RC on refs

proc isIsolated*[T: ref](v: T): bool =
  let p = cast[pointer](v)
  if p != nil:
    let rc = head(p).rc shr rcShift
    #echo "- ref ", v.repr, ", RC: ", rc
    # TODO: naive check
    if rc > 0:
      false
    else:
      isIsolated(v[])
  else:
    true


proc verifyIsolated*[T:ref](v: T) =
  if not isIsolated(v):
    raise newException(NotIsolatedError, v.repr)
