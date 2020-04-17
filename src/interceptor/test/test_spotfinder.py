"""
Author      : Lyubimov, A.Y.
Created     : 04/17/2020
Last Changed: 04/17/2020
Description : Unit test for spotfinding on ZMQ-formatted data (from file)
"""

import os
import shutil

from interceptor.test.test_import import TestReader
from interceptor.connector.processor import FastProcessor
from interceptor.command_line.connector_run import parse_command_args

args, _ = parse_command_args().parse_known_args()


def test_spotfinder():
    ''' Tests FastProcessor in spotfinding mode with default settings
    :return: None
    '''

    # Import test ZMQ file
    importer = TestReader()
    data, info = importer.run()

    # Create dummy file for format class
    filename = 'eiger_test_1.stream'
    with open(filename, 'w') as ef:
        ef.write('EIGERSTREAM')

    # generate processor and run spotfinding (default)
    processor = FastProcessor()
    info = processor.run(data, filename, info)

    # cleanup temp files and folders
    os.remove('eiger_test_1.stream')
    shutil.rmtree('debug')

    # Test that frame imported correctly (frame index should be neither -1 nor -999)
    assert int(info['frame_idx']) > 0

    # Check that no errors were recorded
    for key, value in info.items():
        if 'error' in key:
            assert not value

    # Check that spots were found
    assert int(info['n_spots']) > 0

    # Check that correct number of spots was found (fails if spotfinding algorithm
    # changes or if I change defaults settings)
    try:
        assert int(info['n_spots']) == 641
    except AssertionError as e:
        print ("WARNING: {} spots found instead of 641".format(info['n_spots']))
        raise e


if __name__ == "__main__":
    test_spotfinder()

# -- end
