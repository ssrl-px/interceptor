from __future__ import absolute_import, division, print_function

'''
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 03/31/2020
Description : Streaming stills processor for live data analysis
'''

import time

from connector.processor import FastProcessor
from connector.stream import ZMQStream
from dxtbx.format import FormatEigerStream as EigerStream
from dxtbx.model.experiment_list import ExperimentListFactory


class ConnectorBase():
  """ Base class for ZMQReader and ZMQCollector classes """
  def __init__(self, comm, args, name='zmq_thread'):
    ''' Constructor
    :param comm: mpi4py communication instance
    :param args: command line arguments
    :param name: thread name (for logging mostly)
    '''
    self.name = name
    self.comm = comm
    self.rank = comm.Get_rank()  # each process in MPI has a unique id
    self.size = comm.Get_size()  # number of processes running in this job
    self.stop = False
    self.timeout_start = None
    self.args = args

  def generate_processor(self, args):
    processor = FastProcessor(last_stage=args.last_stage)
    return processor

  def initialize_process(self):
    # Generate params, args, etc. if process rank id = 0
    if self.rank == 0:
      # args, _ = self.generate_args()
      processor  = self.generate_processor(self.args)
      info = dict(processor=processor,
                  args=self.args,
                  host=self.args.host,
                  port=self.args.port)
    else:
      info = None

    # send info dict to all processes
    info = self.comm.bcast(info, root=0)

    # extract info
    self.processor = info['processor']
    self.args      = info['args']
    self.host      = info['host']
    self.port      = info['port']

  def signal_stop(self):
    self.stop = True
    self.comm.bcast(self.stop, root=0)

  def timeout(self, reset=False):
    if reset:
      self.timeout_start = None
    else:
      if self.timeout_start is None:
        self.timeout_start = time.time()
      else:
        if time.time() - self.timeout_start >= 30:
          print ('\n****** TIMED OUT! ******')
          exit()


class Reader(ConnectorBase):
  ''' ZMQ Reader: requests a single frame (multipart) from the Eiger,
      converts to dictionary format, attaches to a special format class,
      carries out processing, and sends the result via ZMQ connection to the
      Collector class.
  '''
  def __init__(self, name='zmq_reader', comm=None, args=None):
    super(Reader, self).__init__(name=name, comm=comm, args=args)
    self.initialize_process()

  def make_experiments(self, filename, data):
    EigerStream.injected_data = data
    exp = ExperimentListFactory.from_filenames([filename])
    return exp

  def convert_from_stream(self, frames):
    if len(frames) <= 2:
      if self.args.header and self.args.htype in str(frames[0].bytes):
        self.make_header(frames)
        return None
    else:
      if not self.args.header:
        hdr_frames = frames[:2]
        img_frames = frames[2:]
        if not hasattr(self, 'header'):
          self.make_header(frames=hdr_frames)
      else:
        img_frames = frames
      frame_string = str(img_frames[0].bytes[:-1])[3:-2] # extract dict entries
      frame_split = frame_string.split(',')
      idx = -1
      run_no = -1
      for part in frame_split:
        if 'series' in part:
          run_no = part.split(':')[1]
        if 'frame' in part:
          idx = part.split(':')[1]
      return [run_no, idx, img_frames]

  def make_header(self, frames):
    self.header = [frames[0].bytes[:-1], frames[1].bytes[:-1]]

  def make_data_dict(self, frames):
    frames = self.convert_from_stream(frames)
    if frames is None:
      return [-1, -1, None, 'DATA ERROR: Frames is NONE']

    run_no = frames[0]
    idx = frames[1]
    data = {'header1': self.header[0], 'header2': self.header[1]}
    for frm in frames[2]:
      i = frames[2].index(frm) + 1
      key = 'streamfile_{}'.format(i)
      if i != 3:
        data[key] = frm.bytes[:-1]
      else:
        data[key] = frm.bytes
    msg = 'DATA: Converted successfully!'
    return [run_no, idx, data, msg]

  def process(self, info, frame, filename):
    start = time.time()
    try:
      experiments = self.make_experiments(filename, frame)
    except Exception as exp:
      print ("CONNECTOR ERROR: Could not create ExperimentList object.\n  {}"
             "".format(exp))
      experiments = None
    if experiments:
      info = self.processor.run(experiments=experiments, info=info)
    else:
      info['prc_error'] = 'EXPERIMENT ERROR: ExperimentList a NoneType object'
    info['proc_time'] = time.time() - start
    return info

  def run(self):
    # Write eiger_#.stream file
    eiger_idx = self.rank
    filename = 'eiger_{}.stream'.format(eiger_idx)
    self.name = 'ZMQ_{:03d}'.format(eiger_idx)
    with open(filename, "w") as fh:
      fh.write('EIGERSTREAM')

    # Initialize ZMQ stream listener
    self.stream = ZMQStream(
      name = self.name,
      host=self.host,
      port=self.port,
      socket_type=self.args.stype)

    # intra-process communication (not using MPI... yet)
    uplink = ZMQStream(
      name='{}_uplink'.format(self.name),
      host='localhost',
      port=7121,
      socket_type='push')

    # Start listening for ZMQ stream
    while True:
      start = time.time()
      try:
        fstart = time.time()
        if self.args.stype.lower() == 'req':
          self.stream.send(b"Hello")
        frames = self.stream.receive()
        fel = time.time() - fstart
      except Exception as exp:
        print ('DEBUG: {} CONNECT FAILED! {}'.format(self.name, exp))
        continue
      if self.args.test:
        for idx, frame in enumerate(frames):
          if len(frame.bytes) < 10000:
            print ('debug: {}: {}'.format(idx, frame.bytes))
      else:
        data = self.make_data_dict(frames)
        info = {
          'proc_name' : self.name,
          'run_no'    : data[0],
          'frame_idx' : data[1],
          'beamXY'    : (0, 0),
          'dist'      : 0,
          'n_spots': 0,
          'hres': 99.0,
          'n_indexed': 0,
          'sg': 'NA',
          'uc': 'NA',
          'spf_error': '',
          'idx_error': '',
          'rix_error': '',
          'img_error': '',
          'prc_error': ''
        }
        if data[1] == -1:
          info['img_error'] = data[3]
        else:
          info = self.process(info, frame=data[2], filename=filename)
        elapsed = time.time() - start
        time_info = {
          'total_time'   : elapsed,
          'receive_time' : fel
        }
        info.update(time_info)
        uplink.send_json(info)

      # If frame processing fits within specified interval, sleep for the
      # remainder of that interval; otherwise (or if args.interval == 0) don't
      # sleep at all
      interval = self.args.interval - elapsed
      if interval > 0:
        time.sleep(interval)

      if self.stop:
        break
    self.stream.close()

  def abort(self):
    self.stop = True


class Collector(ConnectorBase):
  ''' Runs as 0-ranked process in MPI; collects all processing results from
      individual Reader processes and prints them to stdout and sends them
      off as a single stream to the UI if requested.
  '''
  def __init__(self, name='ZMQ_000', comm=None, args=None):
    super(Collector, self).__init__(name=name, comm=comm, args=args)
    self.initialize_process()

  def run(self):
    # todo: decide whether to leave this port hardcoded
    host = 'localhost'
    port = 7121
    socket_type = 'pull'
    collector = ZMQStream(
      name=self.name,
      host=host,
      port=port,
      socket_type=socket_type,
      bind=True)

    send_to_ui = self.args.send or (self.args.uihost and self.args.uiport)
    if send_to_ui:
      ui_stream = ZMQStream(
        name=self.name + "_uplink",
        host=self.args.uihost,
        port=self.args.uiport,
        socket_type=self.args.uistype)

    while True:
      info = collector.receive_json()
      if info:
        try:
          # message to DHS / UI
          prefix = 'htos_log note zmaDhs'
          err_list = [
            info['img_error'],
            info['spf_error'],
            info['idx_error'],
            info['rix_error'],
            info['prc_error']
          ]
          errors = ';'.join([i for i in err_list if i != ''])
          ui_msg = '{0} {1} {2} {3} {4} ' \
                   '{5} {6:.2f} {7} {8} {9} {{{10}}}' \
                   ''.format(
            prefix,             # required for DHS logging
            info['run_no'],     # run number
            info['frame_idx'],  # frame index
            info['n_spots'],    # number_of_spots
            0,                  # TODO: number_of_spots_with_overloaded_pixels
            info['n_indexed'],  # number of indexed reflections
            info['hres'],       # high resolution boundary
            0,                  # TODO: number_of_ice-rings
            info['sg'],         # space group
            info['uc'],         # unit cell
            errors,             # errors
          )
        except Exception as exp:
          print('PRINT ERROR: ', exp)
        else:
          if self.args.verbose:
            print ('*** ({}) RUN {}, FRAME {}:'.format(
              info['proc_name'], info['run_no'], info['frame_idx']))
            print ("  {}".format(ui_msg))
            print ("  DEBUG: BEAM X = {:.2f}, Y = {:.2f}, DIST = {:.2f}".format(
              info['beamXY'][0], info['beamXY'][1], info['dist']
            ))
            print ('  TIME: recv = {:.2f} sec, proc = {:.2f} sec,'
                   ' total = {:.2f} sec'.format(
              info['receive_time'], info['proc_time'], info['total_time']))
            print ('***\n')

          # send string to UI (DHS and/or Interceptor GUI)
          if send_to_ui:
            try:
              ui_stream.send_string(ui_msg)
            except Exception as exp:
              pass
