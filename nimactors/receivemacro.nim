
import std/macros
import std/tables

import ../nimactors


proc genMatch(match: NimNode, code: NimNode, gotMatch: NimNode, captures: seq[NimNode]): NimNode =

  var filter: NimNode
  let fn = genSym(nskProc, "fn")
  let msg = ident("msg")
  var kind: NimNode
  var matchKeys: Table[string, string]

  if match.kind == nnkObjConstr:
    kind = match[0]
    filter = quote:
      `msg` of `kind`
    for i in 1..<match.len:
      match[i].expectKind(nnkExprColonExpr)
      let key = match[i][0]
      let val = match[i][1]
      if val notin captures:
        filter = quote:
          `filter` and `msg`.`kind`.`key` == `val`
      else:
        matchKeys[$val] = $key
  else:
    kind = match
    filter = quote:
      `msg` of `kind`

  let lets = nnkLetSection.newTree()
  for capture in captures:
    let r = ident(matchKeys[$capture])
    let tmp = quote:
      `msg`.`r`
    lets.add newIdentDefs(capture, newEmptyNode(), tmp)

  let n = quote do:
    proc `fn`(`msg`: Message): bool =
      `filter`
    let msg = tryRecv(`fn`).`kind`
    if msg != nil:
      let `msg` = msg
      `lets`
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
      o.add genMatch(nc[0], nc[1], gotMatch, @[])
    elif nc.kind == nnkAsgn:
      var captures: seq[NimNode]
      if nc[0].kind == nnkTupleConstr:
        for nnc in nc[0]:
          captures.add nnc
      else:
        captures.add nc[0]
      o.add genMatch(nc[1][0], nc[1][1], gotMatch, captures)

  o.add quote do:
    if not `gotMatch`:
      suspend()

  echo o.repr
  o
