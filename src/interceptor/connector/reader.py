
from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 05/06/2020
Last Changed: 05/06/2020
Description : Reader module
"""

import os
import time

from mpi4py import MPI

from interceptor.connector import stream
from interceptor.command_line.connector_run import parse_command_args

def initalize_sockets(args, wid):
    rport = "6{}".format(str(args.port)[1:])
    r_socket = stream.make_socket(args.host, rport, socket_type=args.stype, verbose=args.verbose, wid=wid)

    cport = "7{}".format(str(args.port)[1:])
    c_socket = stream.make_socket(args.host, cport, socket_type='push', verbose=args.verbose, wid="{}_2C".format(wid))


def process_stream(args, wid):
    w_socket, c_socket = initalize_sockets(args, wid)

    while True:
        w_socket.send(b"READY")
        addr, empty, frames = w_socket.recv_multipart()
        if frames != b"STANDBY":
            results = drain(frames, wid)
            c_socket.send(results)
        else:
            time.sleep(1)


def drain(frames, wid):
    results = "{} | {}".format(
        str(frames[2].bytes[:-1])[3:-2],
    "({})".format(wid),
    )
    print(results)
    return results


def entry_point():
    args, _ = parse_command_args().parse_known_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    process_stream(args, wid="RDR_{}".format(rank-1))


if __name__ == "__main__":
    entry_point()
