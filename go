#!/bin/sh
make && valgrind --tool=helgrind --quiet ./main
