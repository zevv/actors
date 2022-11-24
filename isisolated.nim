
# TODO: figure out how to support refs in objects, reordering procs
#       and adding fwd declarations causes nim to complain about gcsafe

# Defenitions below stolen from lib/system/arc.nim


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


proc isIsolated*[T: not (ref or seq or array or object or tuple)](v: T): bool =
  true


proc isIsolated*[T: seq or array](vs: T): bool =
  for v in vs:
    if not isIsolated(v):
      return false
  true


proc isIsolated*[T: (object or tuple) and not ref](v: T): bool =
  for k, v in fieldPairs(v):
    let iso = isIsolated(v)
    #echo "- ", k, " ", typeof(v), " ", iso
    if not iso:
      return false
  true


proc isIsolated*[T: ref](v: T): bool =
  let p = cast[pointer](v)
  if not p.isNil:
    let rc = head(p).rc shr rcShift
    if rc > 0:
      false
    else:
      isIsolated(v[])
  else:
    true

proc assertIsolated*[T:ref](v: T) =
  let p = cast[pointer](v)
  if p != nil:
    let rc = head(p).rc shr rcShift
    assert rc == 0
  when false:
    {.cast(gcsafe).}: # whiner
      doassert isIsolated(v)
