
const
  usesValgrind {.booldefine.} = false

when usesValgrind:
  const header = "<valgrind/helgrind.h>"

  proc valgrind_annotate_happens_before*(x: pointer) {.
    header: header, importc: "ANNOTATE_HAPPENS_BEFORE".}
  proc valgrind_annotate_happens_after*(x: pointer) {.
    header: header, importc: "ANNOTATE_HAPPENS_AFTER".}
  proc valgrind_annotate_happens_before_forget_all*(x: pointer) {.
    header: header, importc: "ANNOTATE_HAPPENS_BEFORE_FORGET_ALL".}

  let enabled {.header: header, importc: "RUNNING_ON_VALGRIND".}: bool

  proc running_on_valgrind*(): bool =
    {.cast(noSideEffect), cast(gcSafe).}:
      result = enabled

else:
  proc valgrind_annotate_happens_before*(x: pointer) = discard
  proc valgrind_annotate_happens_after*(x: pointer) = discard
  proc valgrind_annotate_happens_before_forget_all*(x: pointer) = discard

  template running_on_valgrind*(): bool = false
