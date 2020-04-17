"""
Author      : Lyubimov, A.Y.
Created     : 04/06/2020
Last Changed: 04/16/2020
Description : Unit test for ZMQ format import (from file)
"""

import os
from interceptor import packagefinder

from interceptor.connector.connector import Reader
from interceptor.command_line.connector_run import parse_command_args
args, _ = parse_command_args().parse_known_args()


class TestReader(Reader):
    def __init__(self, name='test'):
        super(TestReader, self).__init__(name=name, args=args)

    def run(self):
        frames = []
        img_dir = packagefinder('images', 'test', module='interceptor')
        for n in range(6):
            filename = 'zmq_000001_{:02d}.zmq'.format(n+1)
            filepath = os.path.join(img_dir, filename)
            with open(filepath, 'rb') as fh:
                frames.append(fh.read())

        data = self.make_data_dict(frames)
        return data


def test_reader_from_file():
    reader = TestReader()
    run_no, idx, data, msg = reader.run()

    # check that the data keys line up with correct entries
    # datasest header
    assert 'header_detail' in str(data['header1'])
    assert 'detector_number' in str(data['header2'])
    # frame header
    assert 'frame' in str(data['streamfile_1'])
    assert 'shape' in str(data['streamfile_2'])
    # frame footer
    assert 'stop_time' in str(data['streamfile_4'])

    # check that 'streamfile_3' contains a large string (i.e. image)
    assert len(data['streamfile_3']) > 100000


if __name__ == '__main__':
    test_reader_from_file()
