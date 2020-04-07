from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 04/06/2020
Last Changed: 04/06/2020
Description : Launches multiple ZMQ Connector instances via MPI
"""

import argparse
from libtbx import easy_run

from interceptor.connector import presets
from iota.components.iota_threads import CustomRun

try:
  import importlib.resources as pkg_resources
except ImportError:
  # Try backported to PY<37 `importlib_resources`.
  import importlib_resources as pkg_resources

def parse_command_args():
  """ Parses command line arguments (only options for now) """
  parser = argparse.ArgumentParser(
    prog='connector_run.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=('ZMQ Stream Connector'),
    epilog=('\n{:-^70}\n'.format('')))
  parser.add_argument(
    '-n', '--n_proc', type=int, nargs='?', default=10,
    help='Number of processors')
  parser.add_argument(
    '--host', type=str, nargs='?', default='localhost',
    help='ZMQ server to listen to')
  parser.add_argument(
    '--port', type=str, nargs='?', default='6677',
    help='Port to listen to')
  parser.add_argument(
    '--stype', type=str, nargs='?', default='req',
    help='Socket type')
  parser.add_argument(
    '--uihost', type=str, nargs='?', default=None,
    help='UI host server to send to')
  parser.add_argument(
    '--uiport', type=str, nargs='?', default=None,
    help='UI port to send to')
  parser.add_argument(
    '--uistype', type=str, nargs='?', default='push',
    help='UI socket type')
  parser.add_argument(
    '--interval', type=float, nargs='?', default='0',
    help='Interval between image receipt')
  parser.add_argument(
    '--t', '--timeout', type=int, nargs='?', default=0,
    help='Timeout in seconds when data not coming')
  parser.add_argument(
    '--last_stage', type=str, nargs='?', default='spotfinding',
    help='"Spotfinding", "indexing", or "integration" works')
  parser.add_argument(
    '--test', action='store_true', default=False)
  parser.add_argument(
    '--verbose', action='store_true', default=False,
    help='Print output to stdout')
  parser.add_argument(
    '--iota', action='store_true', default=False,
    help='Use IOTA Processor')
  parser.add_argument(
    '-b', '--beamline', type=str, nargs='?', default=None,
    help='Beamline filename (e.g. "12-1") will select host and port'
  )
  parser.add_argument(
    '-e', '--experiment', type=str, nargs='?', default=None,
    help='Experiment preset (e.g. "injector") will select number of '
         'processors and extent of processing'
  )
  parser.add_argument(
    '-u', '--ui', type=str, nargs='?', default=None,
    help='UI preset (e.g. "gui") will select to which port and host the output '
         'is sent'
  )
  parser.add_argument(
    '--dry_run', action='store_true', default=False,
    help='Print the full command-line and exit without running')

  return parser


def parse_presets(filename, value):
  preset_string = pkg_resources.read_text(presets, filename + '.cfg')
  preset = None
  contents = None
  for ln in preset_string.splitlines():
    item, info = [i.strip().replace('-', '').lower() for i in ln.split('=')]
    if item == value.strip().replace('-', '').lower():
      settings = info.split(':')
      break
  return settings

def entry_point():
  args, _ = parse_command_args().parse_known_args()

  # parse presets if appropriate
  if args.beamline:
    host, port = parse_presets('beamlines', args.beamline)
  else:
    host = args.host
    port = args.port
  connect_options = '--host {} --port {}'.format(host, port)
  if args.experiment:
    n_proc, last_stage = parse_presets('experiments', args.experiment)
  else:
    n_proc = args.n_proc
    last_stage = args.last_stage
  run_options = '--last_stage {}'.format(last_stage)
  if args.ui:
    uihost, uiport = parse_presets('ui', args.ui)
    ui_options = '--uihost {} --uiport {} --send'.format(uihost, uiport)
  else:
    ui_options = ''

  # assemble launch command
  mpi_cmd = 'mpirun --map-by core --bind-to core -np {}'.format(n_proc)
  cmd = '{0} connector {1} {2} {3}'.format(
    mpi_cmd,
    connect_options,
    run_options,
    ui_options)

  # run mpi
  print (cmd)
  if not args.dry_run:
    # easy_run.fully_buffered(cmd, join_stdout_stderr=True).show_stdout()
    job = CustomRun(command=str(cmd), join_stdout_stderr=True)
    job.run()
    job.show_stdout()

# ---------------------------------------------------------------------------- #


if __name__ == '__main__':
  entry_point()

# -- end
