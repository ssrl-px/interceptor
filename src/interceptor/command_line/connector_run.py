from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 03/31/2020
Description : ZMQ Connector launched with MPI. For example:

To run on 10 cores at host 'bl121proc00', receiving from port 8121, running 
only spotfinding, text output to stdout, not forwarding to a GUI:

mpirun --map-by core --bind-to core -np 10 python connector --host 
bl121proc00 --port 8121 --last_stage spotfinding --verbose

The same run as above, with forwarding to a GUI on port 9998:

mpirun --map-by core --bind-to core -np 10 python connector --host 
bl121proc00 --port 8121 --last_stage spotfinding --verbose 
--uihost=localhost --uiport=9998 --uistype='push'
"""

import argparse
from interceptor.connector.connector import Reader, Collector


def parse_command_args():
    """ Parses command line arguments (only options for now) """
    parser = argparse.ArgumentParser(
        prog="connector_run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("ZMQ Stream Connector"),
        epilog=("\n{:-^70}\n".format("")),
    )
    parser.add_argument(
        "-n", "--n_proc", type=int, nargs="?", default=10, help="Number of processors"
    )
    parser.add_argument(
        "--host",
        type=str,
        nargs="?",
        default="localhost",
        help="ZMQ server to listen to",
    )
    parser.add_argument(
        "--port", type=str, nargs="?", default="6677", help="Port to listen to"
    )
    parser.add_argument(
        "--stype", type=str, nargs="?", default="req", help="Socket type"
    )
    parser.add_argument(
        "--uihost", type=str, nargs="?", default=None, help="UI host server to send to"
    )
    parser.add_argument(
        "--uiport", type=str, nargs="?", default=None, help="UI port to send to"
    )
    parser.add_argument(
        "--uistype", type=str, nargs="?", default="push", help="UI socket type"
    )
    parser.add_argument(
        "--interval",
        type=float,
        nargs="?",
        default="0",
        help="Interval between image receipt",
    )
    parser.add_argument(
        "--t",
        "--timeout",
        type=int,
        nargs="?",
        default=0,
        help="Timeout in seconds when data not coming",
    )
    parser.add_argument(
        "--last_stage",
        type=str,
        nargs="?",
        default="spotfinding",
        help='"Spotfinding", "indexing", or "integration" works',
    )
    parser.add_argument(
        "--phil",
        type=str,
        nargs="?",
        default=None,
        help='Absolute path to file with PHIL settings',
    )
    parser.add_argument("--header", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print output to stdout"
    )
    parser.add_argument(
        "--send", action="store_true", default=False, help="Forward results to GUI"
    )
    parser.add_argument(
        "--iota", action="store_true", default=False, help="Use IOTA Processor"
    )
    parser.add_argument(
        "-b",
        "--beamline",
        type=str,
        nargs="?",
        default=None,
        help='Beamline filename (e.g. "12-1") will select host and port',
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        nargs="?",
        default=None,
        help='Experiment preset (e.g. "injector") will select number of '
        "processors and extent of processing",
    )
    parser.add_argument(
        "-u",
        "--ui",
        type=str,
        nargs="?",
        default=None,
        help='UI preset (e.g. "gui") will select to which port and host the output '
        "is sent",
    )
    parser.add_argument(
        "-r",
        "--record",
        type=str,
        nargs="?",
        default=None,
        help="If filepath is supplied, record proc info into file",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print the full command-line and exit without running",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        default=False,
        help="Measure time per frame and output when run is terminated",
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        default=False,
        help="In True, will not process received images; used for testing",
    )

    return parser


def entry_point():
    try:
        from mpi4py import MPI
    except ImportError as ie:
        comm_world = None
        print("DEBUG: MPI NOT LOADED! {}".format(ie))
    else:
        comm_world = MPI.COMM_WORLD
    if comm_world is not None:
        args, _ = parse_command_args().parse_known_args()
        rank = comm_world.Get_rank()
        if rank == 0:
            script = Collector(comm=comm_world, args=args)
        else:
            script = Reader(comm=comm_world, args=args)
        comm_world.barrier()
        script.run()


if __name__ == "__main__":
    entry_point()

# -- end
