from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 05/06/2020
Last Changed: 05/06/2020
Description : Reader module
"""

import time

from mpi4py import MPI

from interceptor.connector import stream
from interceptor.command_line.connector_run import parse_command_args


def initalize_sockets(args, wid, localhost="localhost"):
    rport = "6{}".format(str(args.port)[1:])
    r_socket = stream.make_socket(
        host=localhost, port=rport, socket_type="req", verbose=args.verbose,
        wid=wid
    )

    cport = "7{}".format(str(args.port)[1:])
    c_socket = stream.make_socket(
        host=localhost,
        port=cport,
        socket_type="push",
        verbose=args.verbose,
        wid="{}_2C".format(wid),
    )
    return r_socket, c_socket


def process_stream(args, wid, localhost="localhost"):
    r_socket, c_socket = initalize_sockets(args, wid, localhost)

    while True:
        r_socket.send(b"READY")
        frames = r_socket.recv_multipart()
        if frames[0] != b"STANDBY":
            results = drain(frames[3:], wid)
            c_socket.send(results.encode('utf-8'))
        else:
            time.sleep(1)


def drain(frames, wid):
    results = "{} | {}".format(str(frames[2][:-1])[3:-2], "({})".format(wid),)
    print(results.encode('utf-8'))
    return results


def entry_point(localhost="localhost"):
    args, _ = parse_command_args().parse_known_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    process_stream(args, wid="RDR_{}".format(rank - 1), localhost=localhost)


if __name__ == "__main__":
    entry_point()
