from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 04/06/2020
Last Changed: 05/15/2020
Description : Launches multiple ZMQ Connector instances via MPI
"""

import time

from interceptor.command_line.connector_run import parse_command_args
from interceptor.connector.connector import Collector

times = []


def make_command_line(args):
    # parse presets if appropriate
    connector_commands = ["interceptor"]
    for arg, value in vars(args).items():
        if value and arg not in ['mpi_bind']:
            if value is True:
                cmd_list = ["--{}".format(arg)]
            else:
                cmd_list = ["--{}".format(arg), value]
            connector_commands.extend(cmd_list)
    return connector_commands


def entry_point():
    args, _ = parse_command_args().parse_known_args()
    if not args.dry_run:
        try:
            from mpi4py import MPI
        except ImportError as ie:
            print("DEBUG: MPI NOT LOADED! {}".format(ie))
            exit()
        else:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            localhost = MPI.Get_processor_name().split('.')[0]
            if rank == 0:
                script = Collector(comm=comm, args=args, localhost=localhost)
                script.run()


# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    entry_point()

# -- end
