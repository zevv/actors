
import times


const optBitline {.strdefine.} = ""


when optBitline != "":

  var fLog = open(optBitline, fmWrite)

  proc log(line: string) =
    fLog.write $epochTime() & " " & line & "\n"
    fLog.flushFile()

  proc log(tag, msg: string) =
    log(tag & " " & msg)

  proc logStart*(tag: string) =
    log("+", tag)

  proc logStop*(tag: string) =
    log("-", tag)

  proc logEvent*(tag: string) =
    log("!", tag)

  proc logValue*[T](tag: string, val: T) =
    log("g", tag & " " & $val)

  template log*(tag: string, code: untyped) =
    logStart(tag)
    code
    logStop(tag)


else:
  
  proc logStart*(tag: string) =
    discard

  proc logStop*(tag: string) =
    discard

  proc logEvent*(tag: string) =
    discard

  proc logValue*[T](tag: string, val: T) =
    discard

  template log*(tag: string, code: untyped) =
    code

