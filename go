#!/bin/sh

set -e

nimflags="-d:danger -d:usemalloc --gc:arc --debugger:native -d:optbitline:bitline.log"

run()
{
	case $1 in
		perf)
			nim c ${nimflags} main.nim && ./main
			;;
		asan)
			nim c ${nimflags} --passC:-fsanitize=thread --passL:-fsanitize=thread main.nim && ./main
			;;
		valgrind)
			nim c ${nimflags}  main.nim  && valgrind --leak-check=full --show-leak-kinds=all ./main
			;;
		helgrind)
			nim c ${nimflags} main.nim  && valgrind --tool=helgrind ./main
			;;
	esac
}

for arg in "$@"; do
	run $arg
done
