#!/bin/sh

set -e

nimflags="-d:usemalloc --gc:arc --debugger:native"

run()
{
	cmd=$1
	src=$2
	bin=`echo $src | sed -e 's/.nim$//g'`

	case $1 in
		run)
			nim c ${nimflags} -d:danger ${src} && ${bin}
			;;
		perf)
			nim c ${nimflags} -d:danger ${src} && perf record -g ${bin}
			;;
		tsan)
			nim c ${nimflags} -d:danger --passC:-fsanitize=thread --passL:-fsanitize=thread ${src} && ${bin}
			;;
		asan)
			nim c ${nimflags} -d:danger --passC:-fsanitize=address --passL:-fsanitize=address ${src} && ${bin}
			;;
		valgrind)
			nim c ${nimflags} -d:danger ${src}  && valgrind --quiet --leak-check=full --show-leak-kinds=all ${bin}
			;;
		helgrind)
			nim c ${nimflags} -d:danger ${src}  && valgrind --quiet --tool=helgrind ${bin}
			;;
		drd)
			nim c ${nimflags} -d:danger ${src}  && valgrind --quiet --tool=drd --suppressions=./misc/valgrind-drd-suppressions ${bin}
			;;
		bitline)
			nim c ${nimflags} -d:danger -d:optbitline:bitline.log ${src} && ${bin}
			;;
		debug)
			nim c ${nimflags} ${src} && ${bin}
			;;
	esac
}

run $1 $2
