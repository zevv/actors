#!/bin/sh

set -e

nimflags="-d:danger -d:usemalloc --gc:arc --debugger:native"

run()
{
	case $1 in
		run)
			nim c ${nimflags} main.nim && ./main
			;;
		perf)
			nim c ${nimflags} main.nim && perf record -g ./main
			;;
		asan)
			nim c ${nimflags} --passC:-fsanitize=thread --passL:-fsanitize=thread main.nim && ./main
			;;
		asan)
			nim c ${nimflags} --passC:-fsanitize=thread --passL:-fsanitize=thread main.nim && ./main
			;;
		valgrind)
			nim c ${nimflags} main.nim  && valgrind --quiet --leak-check=full --show-leak-kinds=all ./main
			;;
		helgrind)
			nim c ${nimflags} main.nim  && valgrind --quiet --tool=helgrind ./main
			;;
		bitline)
			nim c ${nimflags} -d:optbitline:bitline.log main.nim && ./main
			;;
		all)
			./go asan valgrind helgrind bitline
			;;
	esac
}

for arg in "$@"; do
	run $arg
done
