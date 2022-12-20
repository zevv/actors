
{.emit: "#include <valgrind/valgrind.h>".}

proc running_on_valgrind*(): bool =
  {.emit: """
    result = RUNNING_ON_VALGRIND;
  """.}

