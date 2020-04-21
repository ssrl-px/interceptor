"""
Author      : Lyubimov, A.Y.
Created     : 04/20/2020
Last Changed: 04/20/2020
Description : pytest fixtures for data importing and processing
"""

import os
import shutil
import pytest

from interceptor import packagefinder
from interceptor.connector.connector import Reader
from interceptor.connector.processor import FastProcessor
from interceptor.command_line.connector_run import parse_command_args
args, _ = parse_command_args().parse_known_args()


class MinimalReader(Reader):
    def __init__(self, name='test'):
        super(MinimalReader, self).__init__(name=name, args=args)

    def run(self):
        frames = []
        img_dir = packagefinder('images', 'test', module='interceptor')
        for n in range(6):
            filename = 'zmq_000001_{:02d}.zmq'.format(n + 1)
            filepath = os.path.join(img_dir, filename)
            with open(filepath, 'rb') as fh:
                frames.append(fh.read())

        data = self.make_data_dict(frames)
        return data


@pytest.fixture(scope="module")
def imported_data():
    reader = MinimalReader()
    return reader.run()


@pytest.fixture(scope='module')
def proc_for_testing():
    reader = MinimalReader()

    argstring = '-b 12-1 -e mesh'.split()
    test_args, _ = parse_command_args().parse_known_args(argstring)

    processor = reader.generate_processor(test_args)
    return processor


@pytest.fixture(scope="module")
def process_test_image(imported_data, proc_for_testing):
    data, info = imported_data

    # Create dummy file for format class
    filename = 'eiger_test_1.stream'
    with open(filename, 'w') as ef:
        ef.write('EIGERSTREAM')

    # run spotfinding (default)
    info = proc_for_testing.run(data, filename, info)

    # cleanup temp files and folders
    os.remove('eiger_test_1.stream')
    shutil.rmtree('debug')

    return info
