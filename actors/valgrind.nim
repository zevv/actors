
{.emit:"#include <valgrind/helgrind.h>".}

template valgrind_annotate_happens_before*(x) =
  block:
    let y {.exportc,inject.} = x
    {.emit:"ANNOTATE_HAPPENS_BEFORE(y);".}

template valgrind_annotate_happens_after*(x) =
  block:
    let y {.exportc,inject.} = x
    {.emit:"ANNOTATE_HAPPENS_AFTER(y);".}

template valgrind_annotate_happens_before_forget_all*(x) =
  block:
    let y {.exportc,inject.} = x
    {.emit:"ANNOTATE_HAPPENS_BEFORE_FORGET_ALL(y);".}

proc running_on_valgrind*(): bool =
  {.emit: "result = RUNNING_ON_VALGRIND;".}
