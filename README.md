
> "_“moar threads of computation”_"

This is an experimental project to create a threaded, share-nothing actor based
framework on top of CPS

This module consists of various parts that will likely be split off at a later time:

- `pkg/nimactors`: the core threadpool + actor implementation
- `pkg/nimactors/lib/evq`: linux epoll based event queue on top of nimactors
- `pkt/nimactors/lib/asyncio`: provides basic async I/O functions on top of the evq


Inspiration was taken from Go channels and Elixir processes

- https://www.erlang.org/doc/reference_manual/processes.html


