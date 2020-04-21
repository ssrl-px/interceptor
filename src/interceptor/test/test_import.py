"""
Author      : Lyubimov, A.Y.
Created     : 04/16/2020
Last Changed: 04/20/2020
Description : Unit test for ZMQ format import (from file)
"""

from interceptor.command_line.connector_run import parse_command_args
args, _ = parse_command_args().parse_known_args()


def test_zmq_import_header(imported_data):
    data, info = imported_data
    # check that the data keys line up with correct entries
    # datasest header
    assert 'header_detail' in str(data['header1'])
    assert 'detector_number' in str(data['header2'])


def test_zmq_import_frame(imported_data):
    data, info = imported_data
    # frame header
    assert 'frame' in str(data['streamfile_1'])
    assert 'shape' in str(data['streamfile_2'])
    # frame footer
    assert 'stop_time' in str(data['streamfile_4'])


def test_zmq_import_image(imported_data):
    data, info = imported_data
    # check that 'streamfile_3' contains a large string (i.e. image)
    assert len(data['streamfile_3']) > 100000

