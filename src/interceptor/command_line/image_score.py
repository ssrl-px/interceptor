from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 05/16/2022
Last Changed: 05/16/2022
Description : File processor will process single image
"""

import os
import argparse
from time import time
from interceptor import __version__ as intxr_version
from interceptor import packagefinder, read_config_file
from interceptor.connector.processor import FileProcessor
from interceptor.connector.fp_worker import FastProcessor
from interceptor.connector.ai_worker import AIProcessor
from interceptor.connector import make_result_string, print_to_stdout

def parse_command_args():
    """ Parses command line arguments (only options for now) """
    parser = argparse.ArgumentParser(
        prog="intxr.score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("ZMQ Stream Connector"),
        epilog=("\n{:-^70}\n".format("")),
    )
    parser.add_argument(
        "path",
        type=str,
        nargs=1,
        default=None,
        help="Path to image file"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Interceptor v{}".format(intxr_version),
        help="Prints version info of Interceptor",
    )
    parser.add_argument(
        "-b",
        "--beamline", type=str, default='DEFAULT', help="Beamline of the experiment",
    )
    parser.add_argument(
        "-c",
        "--processing_config",
        type=str,
        default=None,
        help='Provide a processing config filepath',
    )
    parser.add_argument(
        "-m",
        "--processing_mode",
        type=str,
        default='file',
        help='Provide processing run mode, e.g. "spotfinding", "indexing", etc.',
    )
    parser.add_argument(
        "-s",
        "--series",
        type=int,
        default=-1,
        help='Series number',
    )
    parser.add_argument(
        "-f",
        "--frame",
        type=int,
        default=-1,
        help='Frame number',
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help='Mapping string',
    )
    parser.add_argument(
        "--exposure_time",
        type=float,
        default=0.1,
        help='Exposure time',
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help='Verbose output',
    )

    return parser


def entry_point():
    start = time()

    args, _ = parse_command_args().parse_known_args()
    if not args.path:
        print ("A path to image file(s) needs to be provided")
    else:
        if os.path.isfile(args.path[0]):
            paths = args.path
        elif os.path.isdir(args.path[0]):
            datadir = args.path[0]
            paths = [os.path.join((os.path.abspath(datadir)), f) for f in os.listdir(datadir)]
        else:
            print(f"{args.path} is not a valid path")

    # determine processing config file
    if os.path.isfile(args.processing_config):
        config = read_config_file(args.processing_config)
    else:
        config = packagefinder('processing.cfg', 'connector', read_config=True)
    try:
        cfg = config[args.beamline]
    except KeyError:
        cfg = config['DEFAULT']
    p_configfile = cfg['processing_config_file']

    # initialize processor
    init_time = time()
    if args.processing_mode == 'ai':
        processor = AIProcessor(
            run_mode=args.processing_mode,
            configfile=p_configfile,
            verbose=args.verbose,
        )
    elif args.processing_mode == 'fast':
        processor = FastProcessor(
            run_mode=args.processing_mode,
            configfile=p_configfile,
            verbose=args.verbose,
        )
    else:
        processor = FileProcessor(
            run_mode=args.processing_mode,
            configfile=p_configfile,
            verbose=args.verbose,
        )
    print(f"INITIALIZED PROCESSOR IN {time() - init_time:0.4f} seconds")

    for img_path in paths:
        # initialize info dictionary object
        init_info = {
            "filename": os.path.basename(img_path),
            "full_path": img_path,
            "proc_name": "{}_1".format(args.beamline),
            "wait_time": 0,
            "receive_time": 0,
            "proc_time": 0,
            "total_time": 0,
            "series": args.series,
            "frame": args.frame,
            "run_mode": args.processing_mode,
            "exposure_time": 0.1,
            "mapping": "",
            "sg": None,
            "uc": None,
        }

        print ("PROCESSING {}".format(args.path[0]))

        # Run processor
        info = processor.run(info=init_info, filename=img_path)
        info['total_time'] += info['proc_time']

        # assemble output and print to stdout
        ui_msg = make_result_string(info, cfg)
        print_to_stdout(counter=0, info=info, ui_msg=ui_msg, clip=True)

    print (f"TOTAL TIME = {time() - start:.4f} seconds\n*****\n\n")

if __name__ == "__main__":
	entry_point()

# --> end
