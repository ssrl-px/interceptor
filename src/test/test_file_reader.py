"""
Author      : Lyubimov, A.Y.
Created     : 04/06/2020
Last Changed: 04/06/2020
Description : A test script for ZMQ Reader, which reads in files
"""

import os
import time
import argparse

from iotbx import phil as ip
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex

from interceptor.format import FormatEigerStreamSSRL
from interceptor.connector.processor import FastProcessor

from iota.components.iota_utils import Capturing

# Custom PHIL for spotfinding only
from dials.command_line.find_spots import phil_scope as spf_scope

spf_params_string = """
spotfinder {
  threshold {
    use_trusted_range = False
    algorithm = *dispersion dispersion_extended
    dispersion {
      gain = 1
    }
  }
}
"""

help_message = """
This is a script that will test and time spotfinding and ZMQ decoding. Instead of 
connecting to a ZMQ URL, all information will be read from files. Test files are 
included in ./images, while if tinkering with spotfinding settings is desired, 
a default PHIL file, spf.phil, is included in ./resources.

Example tests:

1. Read in and process a single image once:

dials.python ./src/test/test_file_reader.py ./src/test/images/ --prefix 'zmq' 
--number 000001 --extension zmq --verbose

2. Process five images directly with flex

dials.python ./src/test/test_file_reader.py ./src/test/images/ --prefix 'zmq' 
--number 000001 --extension zmq --repeat 5 --flex --verbose

3. Process five images in parallel with MPI and use timeit to time the processing

dials.python ./src/test/test_file_reader.py ./src/test/images/ --prefix 'zmq' 
--number 000001 --extension zmq --repeat 5 --timeit --mpi --verbose

"""

def parse_test_args():
    """ Parses command line arguments (only options for now) """
    parser = argparse.ArgumentParser(
        prog="test_file_reader.py", description=help_message,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("path", type=str, nargs="?", help="Path to test files")
    parser.add_argument(
        "--phil",
        type=str,
        nargs="?",
        default=None,
        help="Processing parameter file (for flex only)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        nargs="?",
        default="zmq",
        help="Filename prefix (as in <prefix>_<number>_<part>",
    )
    parser.add_argument(
        "--number",
        type=str,
        nargs="?",
        default="000001",
        help="Image number (as in <prefix>_<number>_<part>",
    )
    parser.add_argument(
        "--extension", type=str, nargs="?", default="", help="File extension"
    )
    parser.add_argument(
        "--last_stage",
        type=str,
        nargs="?",
        default="spotfinding",
        help='"Spotfinding", "indexing", or "integration" works',
    )
    parser.add_argument(
        "--flex",
        action="store_true",
        default=False,
        help="Perform spotfinding dirctly with flex (if False, spotfinding will be "
             "performed by the FastProcessor class)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        nargs="?",
        default=1,
        help="Number of times to repeat the trial in timeit",
    )
    parser.add_argument(
        "--timeit",
        action="store_true",
        default=False,
        help="Use timeit to run the test",
    )
    parser.add_argument(
        "--mpi", action="store_true", default=False, help="Use OpenMPI to run the test"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print output to stdout"
    )
    return parser


def make_phil(phil_file=None, show_phil=False):
    if phil_file:
        with open(phil_file, "r") as pf:
            spf_phil = ip.parse(pf.read())
    else:
        spf_phil = ip.parse(spf_params_string)
    new_phil = spf_scope.fetch(source=spf_phil)
    spf_params = new_phil.extract()

    if show_phil:
        diff_phil = spf_scope.fetch_diff(source=spf_phil)
        diff_phil.show()

    return spf_params


def read_file(args, number=1):
    data = {}
    filepath = "{}_{:06d}".format(args.prefix, number)
    zmq = {
        "header1": "{}_01.{}".format(filepath, args.extension),
        "header2": "{}_02.{}".format(filepath, args.extension),
        "streamfile_1": "{}_03.{}".format(filepath, args.extension),
        "streamfile_2": "{}_04.{}".format(filepath, args.extension),
        "streamfile_3": "{}_05.{}".format(filepath, args.extension),
        "streamfile_4": "{}_06.{}".format(filepath, args.extension),
    }

    filename = "eiger_test_{}.stream".format(number)
    with open(filename, "w") as fh:
        fh.write("EIGERSTREAM")

    for key in zmq:
        fpath = os.path.join(args.path, zmq[key])
        with open(fpath, "rb") as fh:
            item = fh.read()
        if key != "streamfile_3":
            data[key] = item[:-1]
        else:
            data[key] = item

    info = {
        "proc_name": "zmq_test",
        "run_no": 0,
        "frame_idx": 0,
        "beamXY": (0, 0),
        "dist": 0,
        "n_spots": 0,
        "hres": 99.0,
        "n_indexed": 0,
        "sg": "NA",
        "uc": "NA",
        "spf_error": "",
        "idx_error": "",
        "rix_error": "",
        "img_error": "",
        "prc_error": "",
        "comment": "",
    }

    return data, info, filename


def run_proc(args, data, info, filename, number=1):
    processor = FastProcessor(last_stage=args.last_stage)
    spf_params = make_phil(args.phil, show_phil=False)

    with Capturing() as junk:
        FormatEigerStreamSSRL.inject_data(data)
        exp = ExperimentListFactory.from_filenames([filename])

        if args.flex:
            t_start = time.time()
            observed = flex.reflection_table.from_observations(exp, spf_params)
            proc_time = time.time() - t_start
            n_spots = len(observed)
        else:
            t_start = time.time()
            info = processor.run(exp, info)
            proc_time = time.time() - t_start
            n_spots = info["n_spots"]

    if args.verbose:
        print("\n*** channel # {}".format(number))
        print("{} spots found".format(n_spots))
        print("Trial: {}. Time: {:.2f} sec".format(number, proc_time))

    return proc_time


def run_test(args, rank):
    t0 = time.time()
    if args.timeit:
        setup = """
from __main__ import parse_test_args, read_file, run_proc
from iota.components.iota_utils import Capturing
args, _ = parse_test_args().parse_known_args()
data, info, filename = read_file(args)
    """
        stmt = """
run_proc(args, data, info, filename)
      """

        import timeit
        timing = timeit.timeit(setup=setup, stmt=stmt, number=1)
    else:
        data, info, filename = read_file(args)
        timing = run_proc(args, data, info, filename, number=rank)

    info_string = "{:<10.6f}: run #{:<3d} -- {:<3.4f} seconds".format(t0, rank, timing)
    return info_string

def print_phil(phil_file=None):
    print("~~~ Testing ZMQ Reader ~~~")
    print('PHIL Settings for this run: ')
    make_phil(phil_file, show_phil=True)

def entry_point():
    args, _ = parse_test_args().parse_known_args()

    results = []
    if args.mpi:
        from mpi4py import MPI

        comm_world = MPI.COMM_WORLD
        if comm_world is not None:
            rank = comm_world.Get_rank()
            comm_world.barrier()
            if rank == 0 and args.verbose:
                print_phil(args.phil)
            result = run_test(args, rank)
            results.append(result)
    else:
        if args.verbose:
            print_phil(args.phil)
        for i in range(args.repeat):
            result = run_test(args, i)
            results.append(result)

    print ("\n~~~ SUMMARY ~~~")
    for result in results:
        print (result)

# Unit test for ZMQ Reader
if __name__ == "__main__":
    entry_point()

# -- end
