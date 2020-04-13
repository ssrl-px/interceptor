import os
import time
import argparse

from dxtbx.model.experiment_list import ExperimentListFactory

from interceptor.format import FormatEigerStreamSSRL
from interceptor.connector.processor import FastProcessor


def parse_test_args():
  """ Parses command line arguments (only options for now) """
  parser = argparse.ArgumentParser(
    prog='test_file_reader.py',
    description=('Test processor with file'),
  )
  parser.add_argument(
    'path', type=str, nargs='?',
    help='Path to test files')
  parser.add_argument(
    '--prefix', type=str, nargs='?', default='zmq',
    help='Filename prefix (as in <prefix>_<number>_<part>')
  parser.add_argument(
    '--number', type=str, nargs='?', default='000001',
    help='Image number (as in <prefix>_<number>_<part>')
  parser.add_argument(
    '--extension', type=str, nargs='?', default='',
    help='File extension')
  parser.add_argument(
    '--last_stage', type=str, nargs='?', default='spotfinding',
    help='"Spotfinding", "indexing", or "integration" works')
  parser.add_argument(
    '--flex', action='store_true', default=False,
    help='Perform spotfinding with flex'
  )
  return parser


def test_file_reader(args):
  processor = FastProcessor(
    last_stage=args.last_stage,
    test=args.flex
  )
  data = {}
  filepath = '{}_{}'.format(args.prefix, args.number)
  zmq = {
    "header1": '{}_01.{}'.format(filepath, args.extension),
    "header2": '{}_02.{}'.format(filepath, args.extension),
    "streamfile_1": '{}_03.{}'.format(filepath, args.extension),
    "streamfile_2": '{}_04.{}'.format(filepath, args.extension),
    "streamfile_3": '{}_05.{}'.format(filepath, args.extension),
    "streamfile_4": '{}_06.{}'.format(filepath, args.extension),
  }

  filename = 'eiger_test_0.stream'
  with open(filename, "w") as fh:
    fh.write('EIGERSTREAM')

  for key in zmq:
    fpath = os.path.join(args.path, zmq[key])
    with open(fpath, "rb") as fh:
      item = fh.read()
    if key != 'streamfile_3':
      data[key] = item[:-1]
    else:
      data[key] = item

  info = {
    'proc_name': 'zmq_test',
    'run_no': 0,
    'frame_idx': 0,
    'beamXY': (0, 0),
    'dist': 0,
    'n_spots': 0,
    'hres': 99.0,
    'n_indexed': 0,
    'sg': 'NA',
    'uc': 'NA',
    'spf_error': '',
    'idx_error': '',
    'rix_error': '',
    'img_error': '',
    'prc_error': '',
    'comment': '',
  }

  FormatEigerStreamSSRL.inject_data(data)
  exp = ExperimentListFactory.from_filenames([filename])

  start = time.time()
  info = processor.run(exp, info)
  proc_time = time.time() - start

  for key, item in info.items():
    print (key, ' --> ', item)

  print ("PROCESSOR TIME = {:.4f} seconds".format(proc_time))


# Unit test for ZMQ Reader
if __name__ == '__main__':
  print('*** TESTING ZMQ READER ***')

  args, _ = parse_test_args().parse_known_args()
  test_file_reader(args)
