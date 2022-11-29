#!/bin/sh

set -e

nimflags="-d:usemalloc --gc:arc --debugger:native"

run()
{
	case $1 in
		run)
			nim c ${nimflags} -d:danger main.nim && ./main
			;;
		perf)
			nim c ${nimflags} -d:danger main.nim && perf record -g ./main
			;;
		tsan)
			nim c ${nimflags} -d:danger --passC:-fsanitize=thread --passL:-fsanitize=thread main.nim && ./main
			;;
		asan)
			nim c ${nimflags} -d:danger --passC:-fsanitize=address --passL:-fsanitize=address main.nim && ./main
			;;
		valgrind)
			nim c ${nimflags} -d:danger main.nim  && valgrind --quiet --leak-check=full --show-leak-kinds=all ./main
			;;
		helgrind)
			nim c ${nimflags} -d:danger main.nim  && valgrind --quiet --tool=helgrind ./main
			;;
		drd)
			nim c ${nimflags} -d:danger main.nim  && valgrind --quiet --tool=drd --suppressions=./misc/valgrind-drd-suppressions ./main
			;;
		bitline)
			nim c ${nimflags} -d:danger -d:optbitline:bitline.log main.nim && ./main
			;;
		debug)
			nim c ${nimflags} main.nim && ./main
			;;
	esac
}

for arg in "$@"; do
	run $arg
done
