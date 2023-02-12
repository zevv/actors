
import std/macros
import std/tables

import actors


# From "Programming in Erlang, 2nd edition, page 194":
#
# 1. When we enter a receive statement, we start a timer (but only if an after
#    section is present in the expression).
#
# 2. Take the first message in the mailbox and try to match it against Pattern1,
#    Pattern2, and so on. If the match succeeds, the message is removed from
#    the mailbox, and the expressions following the pattern are evaluated.
#
# 3. If none of the patterns in the receive statement matches the first message
#    in the mailbox, then the first message is removed from the mailbox and
#    put into a “save queue.” The second message in the mailbox is then tried.
#    This procedure is repeated until a matching message is found or until all
#    the messages in the mailbox have been examined.
#
# 4. If none of the messages in the mailbox matches, then the process is sus-
#    pended and will be rescheduled for execution the next time a new message
#    is put in the mailbox. When a new message arrives, the messages in the
#    save queue are not rematched; only the new message is matched.
#
# 5. As soon as a message has been matched, then all messages that have
#    been put into the save queue are reentered into the mailbox in the order
#    in which they arrived at the process. If a timer was set, it is cleared.
#
# 6. If the timer elapses when we are waiting for a message, then evaluate the
#    expressions ExpressionsTimeout and put any saved messages back into the
#    mailbox in the order in which they arrived at the process
#
#
# Translate to this:
#
#  var idx = 0
#  while true:
#
#    var msg = getMsg(idx)
#
#    if not msg.isNil:
#
#      if msg of Message1:
#        discard dropMsg(idx)
#        idx = 0
#        echo "message1"
#
#      elif msg of Message2:
#        discard dropMsg(idx)
#        idx = 0
#        echo "message2"
#
#      elif msg of Message3:
#        discard dropMsg(idx)
#        idx = 0
#        echo "message3"
#
#      else:
#        # no match found
#        inc idx
#
#    else:
#      # no more messages available
#      suspend()
#

proc genMatch(match: NimNode, code: NimNode, captures: seq[NimNode]): NimNode =

  var filter: NimNode
  let fn = genSym(nskProc, "fn")
  var kind: NimNode
  var matchKeys: Table[string, string]

  if match.kind == nnkObjConstr:
    kind = match[0]
    filter = quote:
      msg of `kind`
    for i in 1..<match.len:
      match[i].expectKind(nnkExprColonExpr)
      let key = match[i][0]
      let val = match[i][1]
      if val notin captures:
        filter = quote:
          `filter` and msg.`kind`.`key` == `val`
      else:
        matchKeys[$val] = $key
  elif match.kind == nnkIdent:
    kind = match
    filter = quote:
      msg of `kind`
  else:
    error("boom", match)

  let lets = nnkLetSection.newTree()

  for capture in captures:
    let r = ident(matchKeys[$capture])
    let tmp = quote:
      msg.`kind`.`r`
    lets.add newIdentDefs(capture, newEmptyNode(), tmp)

  let n = nnkElifBranch.newTree(
    filter,
    quote do:
      dropMsg(idx)
      idx = 0
      `lets`
      `code`
  )

  n


macro receive*(n: untyped) =

  let idx = ident("idx")
  let msg = ident("msg")
  let matches = nnkIfStmt.newTree(
    nnkElifBranch.newTree(newLit(false),nnkDiscardStmt.newTree(newEmptyNode()))
  )

  for nc in n:

    if nc.kind == nnkAsgn:
      var captures: seq[NimNode]
      if nc[0].kind == nnkTupleConstr: # (a, b) = ...
        for nnc in nc[0]:
          captures.add nnc
      else:                            # a = ...
        captures.add nc[0]
      matches.add genMatch(nc[1][0], nc[1][1], captures)
    elif nc.kind == nnkCall:
      matches.add genMatch(nc[0], nc[1], @[])
    else:
      error("boom", nc)

  let e = quote do:
    inc `idx`

  matches.add nnkElse.newTree(e)

  let nout = quote do:
    var `idx` = 0
    while true:
      let `msg` = getMsg(`idx`)
      if not `msg`.isNil:
        `matches`
      else:
        suspend()

  echo nout.repr
  nout
