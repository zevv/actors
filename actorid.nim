
type
  
  ActorId* = distinct int

proc `==`*(x, y: ActorId): bool {.borrow.}

proc `$`*(id: ActorID): string =
  return "#AID<" & $(id.int) & ">"

