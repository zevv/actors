

type
  mallinfo* = object
    arena*: csize_t
    ordblks*: csize_t
    smblks*: csize_t
    hblks*: csize_t
    hblkhd*: csize_t
    usmblks*: csize_t
    fsmblks*: csize_t
    uordblks*: csize_t
    fordblks*: csize_t
    keepcost*: csize_t

proc mallinfo2*(): mallinfo {.importc: "mallinfo2".}
