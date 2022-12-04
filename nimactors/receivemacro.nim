
import std/macros

import ../nimactors


proc genMatch(match: NimNode, code: NimNode, gotMatch: NimNode): NimNode =

  var filter: NimNode
  let fn = ident("fn")
  let msg = ident("msg")
  var kind: NimNode

  if match.kind == nnkObjConstr:
    kind = match[0]
    filter = quote:
      `msg` of `kind`
    for i in 1..<match.len:
      match[i].expectKind(nnkExprColonExpr)
      let key = match[i][0]
      let val = match[i][1]
      filter = quote:
        `filter` and `msg`.`kind`.`key` == `val`
  else:
    kind = match
    filter = quote:
      `msg` of `kind`

  let n = quote do:
    block:
      proc `fn`(`msg`: Message): bool =
        `filter`
      let msg = tryRecv(`fn`).`kind`
      if msg != nil:
        block:
          let `msg` = msg
          `gotMatch` = true
          `code`

  n


macro receive*(n: untyped) =

  var o = newStmtList()
  let gotMatch = ident("gotMatch")

  o.add quote do:
    var `gotMatch`: bool

  for nc in n:
    if nc.kind == nnkCall:
      o.add genMatch(nc[0], nc[1], gotMatch)

  o.add quote do:
    if not `gotMatch`:
      suspend()

  echo o.repr
  o
