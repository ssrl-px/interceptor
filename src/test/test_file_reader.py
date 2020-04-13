import os
import time
import argparse

from iotbx import phil as ip
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex

from interceptor.format import FormatEigerStreamSSRL
from interceptor.connector.processor import FastProcessor


# Custom PHIL for spotfinding only
from dials.command_line.find_spots import phil_scope as spf_scope
spf_params_string = '''
spotfinder {
  threshold {
    use_trusted_range = False
    algorithm = *dispersion dispersion_extended
    dispersion {
      gain = 1
    }
  }
}
'''

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
    '--phil', type=str, nargs='?', default=None,
    help='Processing parameter file (for flex only)')
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


def make_phil(phil_file=None):
  if phil_file:
    with open(phil_file, 'r') as pf:
      spf_phil = ip.parse(pf.read())
  else:
    spf_phil = ip.parse(spf_params_string)
  spf_params = spf_scope.fetch(source=spf_phil).extract()

  diff_phil = spf_scope.fetch(source=spf_phil).show()

  return spf_params


def test_file_reader(args):
  processor = FastProcessor(
    last_stage=args.last_stage,
    # test=args.flex
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

  if args.flex:
    print ("Flex testing")
    spf_params = make_phil(args.phil)
    spf_start = time.time()
    observed = flex.reflection_table.from_observations(
      exp, spf_params)
    spf_time = time.time() - spf_start
    print ("{} reflections found".format(len(observed)))
    print('Spf time: {:.4f} sec'.format(spf_time))
  else:
    print ('FastProcessor testing')
    prc_start = time.time()
    info = processor.run(exp, info)
    proc_time = time.time() - prc_start
    print ("{} reflections found".format(info['n_spots']))
    print('Proc time: {:.4f} sec'.format(proc_time))


# Unit test for ZMQ Reader
if __name__ == '__main__':
  print('*** TESTING ZMQ READER ***')

  args, _ = parse_test_args().parse_known_args()
  test_file_reader(args)
