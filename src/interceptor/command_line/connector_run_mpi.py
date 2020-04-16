from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 04/06/2020
Last Changed: 04/06/2020
Description : Launches multiple ZMQ Connector instances via MPI
"""

import os
import time
import procrunner

from interceptor import import_resources
from interceptor.command_line.connector_run import parse_command_args

presets = import_resources(configs="connector", package="connector")
times = []


def entry_point():
    args, _ = parse_command_args().parse_known_args()

    # parse presets if appropriate
    connector_commands = ["connector"]

    # Beamline preset
    if args.beamline:
        host, port = presets["beamlines"].extract(args.beamline)
    else:
        host = args.host
        port = args.port
    connector_commands.extend(
        ["--host", host, "--port", port, "--stype", "req",]
    )

    # Experiment preset
    if args.experiment:
        n_proc, last_stage = presets["experiments"].extract(args.experiment)
    else:
        n_proc = args.n_proc
        last_stage = args.last_stage
    connector_commands.extend(["--last_stage", last_stage])

    # UI preset
    if args.ui:
        uihost, uiport = presets["ui"].extract(args.ui)
        connector_commands.extend(
            ["--uihost", uihost, "--uiport", uiport, "--uistype", "push",]
        )

    for arg, value in vars(args).items():
        if "--{}".format(arg) not in connector_commands and arg not in [
            "beamline",
            "experiment",
            "ui",
            "n_proc",
        ]:
            if value:
                if value is True:
                    cmd_list = ["--{}".format(arg)]
                else:
                    cmd_list = ["--{}".format(arg), value]
                connector_commands.extend(cmd_list)

    # mpi command
    command = list(
        map(
            str,
            [
                "mpirun",
                "--map-by",
                "core",
                "--bind-to",
                "core",
                "--rank-by",
                "core",
                "--np",
                n_proc,
                *connector_commands,
            ],
        )
    )

    # run mpi
    print(" ".join(command))
    if not args.dry_run:
        start = time.time()
        try:
            if args.time:
                callback = get_total_time
            else:
                callback = None
            result = procrunner.run(
                command, callback_stdout=callback, working_directory=os.curdir
            )
        except KeyboardInterrupt:
            print("\n*** Terminated with KeyboardInterrupt")
            if args.time and times:
                print("*** Total processing time: {:.2f} sec".format(times[-1]))
                print(
                    "*** Rate ({} images): {:.2f} Hz".format(
                        len(times), len(times) / times[-1]
                    )
                )
            print("*** Total runtime: {:.2f} sec".format(time.time() - start))
            print(" ... deleting temporary files...")
            curdir = os.path.abspath(os.curdir)
            temp_files = [
                f for f in os.listdir(curdir) if os.path.splitext(f)[-1] == ".stream"
            ]
            for tfile in temp_files:
                tpath = os.path.join(curdir, tfile)
                os.remove(tpath)
            print("\n~~~ fin ~~~")


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
