from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 04/06/2020
Last Changed: 05/15/2020
Description : Launches multiple ZMQ Connector instances via MPI
"""

import os
import time
import procrunner

from interceptor.command_line.connector_run import parse_command_args

times = []


def make_command_line(args):
    # parse presets if appropriate
    connector_commands = ["interceptor"]
    for arg, value in vars(args).items():
        if value and arg not in ['n_proc', 'mpi_bind']:
            if value is True:
                cmd_list = ["--{}".format(arg)]
            else:
                cmd_list = ["--{}".format(arg), value]
            connector_commands.extend(cmd_list)
    return connector_commands


def entry_point():
    args, _ = parse_command_args().parse_known_args()
    iargs = make_command_line(args)
    if not args.dry_run:
        try:
            from mpi4py import MPI
        except ImportError as ie:
            print("DEBUG: MPI NOT LOADED! {}".format(ie))
            exit()
        else:
            localhost = MPI.Get_processor_name().split('.')[0]
            NPROCS = args.n_proc
            COMMAND = ['interceptor'] * NPROCS
            ARGS = [iargs] * NPROCS
            MAXPROCS = NPROCS
            INFO = [MPI.INFO_NULL] * NPROCS

            print ('DEBUG: SPAWNING {} PROCESSES'.format(NPROCS))

            comm_world = MPI.COMM_WORLD
            children = comm_world.Spawn_multiple(
                COMMAND,
                ARGS,
                MAXPROCS,
                info=INFO,
                root=0,
            )
            children.Barrier()
            children.bcast(localhost, root=0)



def get_total_time(ln):
    if "TIME" in ln:
        if times:
            delta = time.time() - times[0]
            times.append(delta)
        else:
            times.append(time.time())


# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    entry_point()

# -- end
