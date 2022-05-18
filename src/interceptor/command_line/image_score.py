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
from interceptor.connector.processor import FileProcessor

def parse_command_args():
    """ Parses command line arguments (only options for now) """
    parser = argparse.ArgumentParser(
        prog="connector_run.py",
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
        default='spotfinding',
        help='Provide processing run mode, e.g. "spotfinding", "indexing", etc.',
    )
    return parser

def entry_point():
    args, _ = parse_command_args().parse_known_args()
    if not args.path:
        print ("A image filepath needs to be provided")
    elif not os.path.isfile(args.path):
        print ("{} is not a valid filepath".format(args.path))

    # initialize processor
    processor = FileProcessor(
        run_mode=args.processing_mode,
        configfile=args.processing_config
    )

    # initialize info dictionary object
    img_path = os.path.abspath(args.path)
    init_info = {
        "filename": os.path.basename(img_path),
        "full_path": img_path,
        "series": -1,
        "frame": -1,
        "run_mode": args.processing_mode,
        "mapping": "",
    }

    info = processor.run(filename=img_path, info=init_info)

    print("File {} processed! Results: ".format(info['filename']))
    print("  No. spots found :    {}".format(info['n_spots']))
    print("  Image score     :    {}".format(info['score']))
    print("  Error           :    {}".format(info['spf_error']))

# --> end
