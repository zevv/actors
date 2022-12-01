#!/bin/sh

set -e
set -x

nimflags="--verbosity:0 -d:usemalloc --mm:arc --debugger:native"

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
			nim c ${nimflags} -d:danger ${src}  && valgrind --error-exitcode=255 --quiet --leak-check=full --show-leak-kinds=all ${bin}
			;;
		valgrind2)
			nim c ${nimflags} -d:danger ${src}  && valgrind --error-exitcode=255 --quiet ${bin}
			;;
		helgrind)
			nim c ${nimflags} -d:danger ${src}  && valgrind --error-exitcode=255 --quiet --tool=helgrind ${bin}
			;;
		drd)
			nim c ${nimflags} -d:danger ${src}  && valgrind --error-exitcode=255 --quiet --tool=drd --suppressions=./misc/valgrind-drd-suppressions ${bin}
			;;
		bitline)
			nim c ${nimflags} -d:danger -d:optbitline:bitline.log ${src} && ${bin}
			;;
		debug)
			nim c ${nimflags} ${src} && ${bin}
			;;
		all)
			./go valgrind tests/texit
			./go helgrind tests/texit
			./go drd tests/texit
			./go tsan tests/texit
			./go asan tests/texit
			./go debug tests/texit
			./go run tests/tmillions
			figlet "all good"
			;;
	esac
}

run $1 $2
