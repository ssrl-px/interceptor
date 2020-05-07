from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 05/06/2020
Last Changed: 05/06/2020
Description : Broker module
"""

from mpi4py import MPI

from interceptor.connector import stream
from interceptor.command_line.connector_run import parse_command_args


def initialize_ends(args, wid):
    """initializes front- and backend sockets

    Parameters
    ----------
    args : [namespace]
        Command line arguments
    """
    wport = "6{}".format(str(args.port)[1:])
    read_end = stream.make_socket(host=args.host, port=wport, socket_type='router', bind=True, verbose=args.verbose)
    data_end = stream.make_socket(host=args.host, port=args.port, socket_type='pull', verbose=args.verbose)
    return read_end, data_end


def main_broker_loop(args, wid):
    read_end, data_end = initialize_ends(args, wid)

    #initialize workers
    readers = []

    # create poller
    poller = stream.make_poller()

    # register worker-facing end with poller, poll for workers to report as ready
    poller.register(read_end)

    while True:
        sockets = dict(poller.poll())
        if read_end in sockets:
            # handle worker activity
            request = read_end.recv_multipart()
            reader, empty, client = request[:3] # not sure what all of the request is...

            if not readers:
                poller.register(data_end)
            readers.append(reader)

        if data_end in sockets:
            client, empty, frames = data_end.recv_multipart()
            reader = readers.pop()
            read_end.send_multipart([reader, b"", client, b"", frames])
            if not readers:
                poller.unregister(data_end)


def entry_point():
    args, _ = parse_command_args().parse_known_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert rank == 0
    main_broker_loop(args, wid='BROKER')


if __name__ == '__main__':
    entry_point()
