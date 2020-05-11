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
from interceptor.connector.utils import decode_frame_header

def initialize_ends(args, wid, localhost="localhost"):
    """ initializes front- and backend sockets """
    wport = "6{}".format(str(args.port)[1:])
    read_end = stream.make_socket(
        host=localhost,
        port=wport,
        socket_type="router",
        bind=True,
        verbose=args.verbose,
        wid="{}_READ".format(wid),
    )
    data_end = stream.make_socket(
        host=args.host,
        port=args.port,
        socket_type="pull",
        verbose=args.verbose,
        wid="{}_DATA".format(wid),
    )
    return read_end, data_end


def main_broker_loop(args, wid, localhost='localhost'):
    read_end, data_end = initialize_ends(args, wid, localhost)

    # initialize workers
    readers = []

    # create poller
    poller = stream.make_poller()

    # register backend and frontend with poller
    poller.register(read_end, 1)
    poller.register(data_end, 1)

    while True:
        sockets = dict(poller.poll())
        if read_end in sockets:
            # handle worker activity
            request = read_end.recv_multipart()
            reader, empty, ready = request
            readers.append(reader)

        if data_end in sockets:
            frames = data_end.recv_multipart()
            if readers:
                reader = readers.pop()
                rframes = [reader, b"", b"BROKER", b""]
                rframes.extend(frames)
                read_end.send_multipart(rframes)
            else:
                frmdict = decode_frame_header(frames[2])
                print ("WARNING! NO READY READERS! Skipping frame #", frmdict['frame'])


def entry_point(localhost='localhost'):
    args, _ = parse_command_args().parse_known_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert rank == 0
    main_broker_loop(args, wid="BROKER", localhost=localhost)


if __name__ == "__main__":
    entry_point()
