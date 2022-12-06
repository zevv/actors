
# TODO: figure out how to support refs in objects, reordering procs
#       and adding fwd declarations causes nim to complain about gcsafe


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


proc getRc*[T:ref](v: T): int =
  let p = cast[pointer](v)
  if p != nil:
    result = head(p).rc shr rcShift



proc isIsolated*[T: not (ref or seq or array or object or tuple)](v: T): bool =
  #echo "- ", typeof(v)
  true

proc isIsolated*[T: seq or array](vs: T): bool =
  #echo "- ", typeof(vs)
  for v in vs:
    if not isIsolated(v):
      return false
  true

proc isIsolated*[T: (object or tuple) and not ref](v: T): bool =
  #echo "- ", typeof(v)
  for k, v in fieldPairs(v):
    #echo "* ", k
    let iso = isIsolated(v)
    if not iso:
      raise newException(NotIsolatedError, "field '" & k & "' in '" & $typeof(v) & "' is not isolated")
      return false
  true

proc isIsolated*[T: ref](v: T): bool =
  let rc = getRc(v)
  #echo "- ref ", typeof(v), ": ", rc
  if rc > 0:
    false
  else:
    isIsolated v[]

proc assertIsolated*[T:ref](v: T, expected=0) =
  {.cast(gcsafe).}:
    if not isIsolated(v):
      raise newException(NotIsolatedError, "ref of type '" & $typeof(v) & "' is not isolated")
