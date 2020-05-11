from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 04/06/2020
Last Changed: 04/20/2020
Description : Launches multiple ZMQ Connector instances via MPI
"""

import os
import time
import procrunner

from interceptor.command_line.connector_run import parse_command_args
times = []


def make_mpi_command_line(args):
    # parse presets if appropriate
    connector_commands = ["lru"]

    for arg, value in vars(args).items():
        if value and arg not in ['n_proc', 'mpi_bind']:
            if value is True:
                cmd_list = ["--{}".format(arg)]
            else:
                cmd_list = ["--{}".format(arg), value]
            connector_commands.extend(cmd_list)

    # mpi command
    if args.mpi_bind:
        estimated_nproc = 0
        ranges = [i.replace(',', '') for i in args.mpi_bind]
        for r in ranges:
            rng = r.split('-')
            if isinstance(rng, list):
                if len(rng) == 2:
                    start = int(rng[0])
                    end = int(rng[1])
                    estimated_nproc += len(range(start, end+1))
                else:
                    estimated_nproc += 1
            else:
                estimated_nproc += 1
        cpus = ','.join(ranges)
        n_proc = estimated_nproc
        command = list(
            map(
                str,
                [
                    "mpirun",
                    "--cpu-set",
                    cpus,
                    "--bind-to",
                    "cpu-list:ordered",
                    "--np",
                    n_proc+2,
                    *connector_commands,
                ],
            )
        )
    else:
        command = list(
            map(
                str,
                [
                    "mpirun",
                    "--map-by",
                    "socket",
                    "--bind-to",
                    "core",
                    "--np",
                    args.n_proc+2,
                    *connector_commands,
                ],
            )
        )
    return command


def entry_point():
    args, _ = parse_command_args().parse_known_args()
    command = make_mpi_command_line(args)
    # run mpi
    print(" ".join(command))
    if not args.dry_run:
        start = time.time()
        try:
            result = procrunner.run(
                command, working_directory=os.curdir
            )
        except KeyboardInterrupt:
            print("\n*** Terminated with KeyboardInterrupt")


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
