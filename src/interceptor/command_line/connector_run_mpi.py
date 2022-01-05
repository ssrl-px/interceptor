from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 04/06/2020
Last Changed: 10/14/2020
Description : Launches multiple ZMQ Connector instances via MPI
"""

import os
import time
import procrunner

from interceptor.command_line.connector_run import parse_command_args

times = []


def make_mpi_command_line(args):
    # parse presets if appropriate
    connector_commands = ["intxr.connect"]

    for arg, value in vars(args).items():
        if value and arg not in ['n_proc', 'mpi_bind', 'host', 'hostfile']:
            if value is True:
                cmd_list = ["--{}".format(arg)]
            else:
                cmd_list = ["--{}".format(arg), value]
            connector_commands.extend(cmd_list)

    # host and hostfile (for specifying multiple hosts)
    hosting_list = []
    if args.hostfile:
        hostfile = ['--hostfile', args.hostfile]
        hosting_list.extend(hostfile)
    if args.host:
        hosts = ['--host']
        hosts.extend(args.host)
        print ('debug: ', hosts, args.host)
        hosting_list.extend(hosts)
    #hosting = ' '.join(hosting_list) if hosting_list else ''

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
                    estimated_nproc += len(range(start, end + 1))
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
                     "--np",
                    n_proc,
                    *hosting_list,
                    "--report-pid",
                    ".current_process_id",
                    "--enable-recovery",
                    "--cpu-set",
                    cpus,
                    "--bind-to",
                    "cpu-list:ordered",
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
                    '--np',
                    args.n_proc,
                    *hosting_list,
                    "--report-pid",
                    ".current_process_id",
                    "--enable-recovery",
                    "--map-by",
                    "socket",
                    "--bind-to",
                    "core",
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
    try:
        with open('.current_process_id') as pidf:
            cpid = pidf.read()[:-1]
        print("Found Interceptor process with PID {}. Terminating.".format(cpid))
        try:
            kill_cmd = ['kill', '-9', cpid]
            result = procrunner.run(kill_cmd)
            os.remove('.current_process_id')
        except Exception as e:
            print(
                "ERROR: Could not terminate process with PID {}, due to error: {}".format(
                    cpid, e))
    except FileNotFoundError:
        pass

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
            curdir = os.path.abspath(os.curdir)
            temp_files = [
                f for f in os.listdir(curdir) if os.path.splitext(f)[-1] == ".stream"
            ]
            for tfile in temp_files:
                tpath = os.path.join(curdir, tfile)
                os.remove(tpath)

            print("\n*** Terminated with KeyboardInterrupt")
            if args.time and times:
                print("*** Total processing time: {:.2f} sec".format(times[-1]))
                print(
                    "*** Rate ({} images): {:.2f} Hz".format(
                        len(times), len(times) / times[-1]
                    )
                )
            print("*** Total runtime: {:.2f} sec".format(time.time() - start))
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
