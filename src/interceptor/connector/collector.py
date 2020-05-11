from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 05/06/2020
Last Changed: 05/06/2020
Description : Collector module
"""

import os
import time

from mpi4py import MPI

from interceptor.connector import stream
from interceptor.command_line.connector_run import parse_command_args


def collect(args, localhost="localhost"):
    cport = "7{}".format(str(args.port)[1:])
    c_socket = stream.make_socket(
        localhost, cport, socket_type="pull", verbose=args.verbose, wid="COLLECTOR"
    )

    while True:
        info = c_socket.recv()
        print("*** COLLECTOR: {}".format(info))


def entry_point(localhost="localhost"):
    args, _ = parse_command_args().parse_known_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert rank == 1
    collect(args, localhost)


if __name__ == "__main__":
    entry_point()
