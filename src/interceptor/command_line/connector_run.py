from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 04/20/2020
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

from interceptor import __version__ as intxr_version
from interceptor import packagefinder
from interceptor.connector.connector import Connector, Reader, Collector


class ExpandPresets(argparse.Action):
    def __init__(self, option_strings, dest, nargs=1, **kwargs):
        self.filename = kwargs.pop('filename', None)
        if not self.filename:
            raise ValueError("A filename needs to be provided")
        if nargs != 1:
            raise ValueError("Nargs must be equal to one")
        super(ExpandPresets, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        config = packagefinder(self.filename, 'connector', read_config=True)
        for key, value in config[values].items():
            setattr(namespace, key, value)


def parse_command_args():
    """ Parses command line arguments (only options for now) """
    parser = argparse.ArgumentParser(
        prog="connector_run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("ZMQ Stream Connector"),
        epilog=("\n{:-^70}\n".format("")),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Interceptor v{}".format(intxr_version),
        help="Prints version info of Interceptor",
    )
    parser.add_argument(
        "-n", "--n_proc", type=int, default=10, help="Number of processors"
    )
    parser.add_argument(
        "--mpi_bind",
        type=str,
        nargs="*",
        default=None,
        help='List of cpus to which the processes will bind (e.g. "1-10,20-54");'
             ' will supersede the --n_proc value even from preset',
    )
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print output to stdout"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Run debug code"
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default=None,
        help='Provide a beamline-specific startup config filepath',
    )
    parser.add_argument(
        "-b",
        "--beamline", type=str, default='DEFAULT', help="Beamline of the experiment",
    )
    parser.add_argument(
        "-r",
        "--record",
        type=str,
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
        "--broker",
        action="store_true",
        default=False,
        help="Insert a broker in between Readers and Splitter",
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
        localhost = MPI.Get_processor_name().split('.')[0]
        if rank == 0:
                script = Collector(comm=comm_world, args=args, localhost=localhost)
        elif rank == 1:
            if args.broker:
                script = Connector(comm=comm_world, args=args, localhost=localhost)
            else:
                script = Reader(comm=comm_world, args=args, localhost=localhost)
        else:
            script = Reader(comm=comm_world, args=args, localhost=localhost)
        comm_world.barrier()

        script.run()


if __name__ == "__main__":
    entry_point()

# -- end
