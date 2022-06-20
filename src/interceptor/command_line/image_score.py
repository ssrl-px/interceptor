from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 05/16/2022
Last Changed: 05/16/2022
Description : File processor will process single image
"""

import os
import argparse
from interceptor import __version__ as intxr_version
from interceptor import packagefinder, read_config_file
from interceptor.connector.processor import FileProcessor
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

    return parser


def entry_point():
    args, _ = parse_command_args().parse_known_args()
    if not args.path:
        print ("A image filepath needs to be provided")
    elif not os.path.isfile(args.path[0]):
        print ("{} is not a valid filepath".format(args.path))

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
    processor = FileProcessor(
        run_mode=args.processing_mode,
        configfile=p_configfile,
    )

    # initialize info dictionary object
    img_path = os.path.abspath(args.path[0])
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

    # Run processor
    info = processor.run(filename=img_path, info=init_info)
    info['total_time'] += info['proc_time']

    # assemble output and print to stdout
    ui_msg = make_result_string(info, cfg)
    print_to_stdout(counter=0, info=info, ui_msg=ui_msg)

# --> end
