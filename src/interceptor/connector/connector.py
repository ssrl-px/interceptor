from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 05/15/2020
Description : Streaming stills processor for live data analysis
"""

import os
import time
import zmq

import numpy as np

from threading import Thread

from interceptor import packagefinder, read_config_file
from interceptor.connector.processor import ZMQProcessor
from interceptor.connector import utils
from interceptor.connector import make_result_string


def debug_segfault():
    """ Deliberate segfault for debugging purposes """
    import ctypes

    ctypes.string_at(1)


class ZMQProcessBase:
    """ Base class for Connector, Reader, and Collector classes """

    def __init__(self, comm, args, name="zmq_thread", localhost="localhost"):
        """ Constructor
    :param comm: mpi4py communication instance
    :param args: command line arguments
    :param name: thread name (for logging mostly)
    :param localhost: the host of the Collector process (ranked 0)
    """
        self.name = name
        self.comm = comm
        self.localhost = localhost

        if comm:
            self.rank = comm.Get_rank()  # each process in MPI has a unique id
            self.size = comm.Get_size()  # number of processes running in this job
        else:
            self.rank = args.rank
            self.size = args.n_proc

        self.stop = False
        self.timeout_start = None
        self.args = args
        self.generate_config()

    def generate_config(self):
        # generate startup config params
        if self.args.config_file:
            s_config = read_config_file(self.args.config_file)
        else:
            s_config = packagefinder('startup.cfg', 'connector', read_config=True)

        # convert to namedtuple because 1) not easily mutable, 2) can call attributes
        self.cfg = s_config[self.args.beamline]

    @staticmethod
    def make_socket(
            socket_type,
            wid,
            host=None,
            port=None,
            url=None,
            bind=False,
            verbose=False,
    ):
        assert (host and port) or url

        # assemble URL from host and port
        if not url:
            url = "tcp://{}:{}".format(host, port)

        # Create socket
        context = zmq.Context()
        socket = context.socket(getattr(zmq, socket_type.upper()))
        socket.identity = wid.encode('ascii')

        # Connect to URL
        socket.connect(url)
        if verbose:
            print('{} connected to {}'.format(wid, url))

        # Bind to port
        if bind:
            if not port:
                bind_port = url[-4:]
            else:
                bind_port = port
            bind_url = "tcp://*:{}".format(bind_port)
            socket.bind(bind_url)
            if verbose:
                print('{} bound to {}'.format(wid, bind_url))

        return socket

    def broadcast(self, data):
        if self.comm:
            self.comm.bcast(data, root=0)


class Connector(ZMQProcessBase):
    """ A ZMQ Broker class, with a zmq.PULL backend (facing a zmq.PUSH Splitter) and
    a zmq.ROUTER front end (facing zmq.REQ Readers). Is intended to a) get images
    from Splitter and assign each image to the least-recently used (LRU) Reader,
    b) handle start-up / shut-down signals, as well as type-of-processing signals
    from Splitter, c) manage MPI processes, d) serve as a failpoint away from
    Splitter, which is writing data to files """

    def __init__(self, name="CONN", comm=None, args=None, localhost="localhost"):
        super(Connector, self).__init__(
            name=name, comm=comm, args=args, localhost=localhost
        )
        self.initialize_ends()
        self.readers = []

    def initialize_ends(self):
        """ initializes front- and backend sockets """
        wport = "6{}".format(str(self.cfg.getstr('port'))[1:])
        self.read_end = self.make_socket(
            socket_type="router",
            wid="{}_READ".format(self.name),
            host=self.localhost,
            port=wport,
            bind=True,
        )
        self.data_end = self.make_socket(
            socket_type="pull",
            wid="{}_DATA".format(self.name),
            host=self.cfg.getstr('host'),
            port=self.cfg.getstr('port'),
        )
        self.poller = zmq.Poller()

    def connect_readers(self):
        # register backend and frontend with poller
        self.poller.register(self.read_end, zmq.POLLIN)
        self.poller.register(self.data_end, zmq.POLLIN)

        while True:
            sockets = dict(self.poller.poll())
            if self.read_end in sockets:
                # handle worker activity
                request = self.read_end.recv_multipart()
                if not request[0] in self.readers:
                    self.readers.append(request[0])

            if self.data_end in sockets:
                # Receive frames and assign to readers
                frames = self.data_end.recv_multipart()
                if self.readers:
                    reader = self.readers.pop()
                    rframes = [reader, b"", b"BROKER", b""]
                    rframes.extend(frames)
                    self.read_end.send_multipart(rframes)
                else:
                    frmdict = utils.decode_frame_header(frames[2])
                    print(
                        "WARNING! NO READY READERS! Skipping frame #", frmdict["frame"]
                    )

    def run(self):
        self.connect_readers()


class Reader(ZMQProcessBase):
    """ ZMQ Reader: requests a single frame (multipart) from the Eiger,
      converts to dictionary format, attaches to a special format class,
      carries out processing, and sends the result via ZMQ connection to the
      Collector class.
  """

    def __init__(self, name="zmq_reader", comm=None, args=None, localhost="localhost"):
        super(Reader, self).__init__(
            name=name, comm=comm, args=args, localhost=localhost
        )
        self.name = "ZMQ_{:03d}".format(self.rank)
        self.generate_processor()

        # Initialize ZMQ sockets
        self.initialize_zmq_sockets()

    def generate_processor(self, run_mode='DEFAULT'):
        self.processor = ZMQProcessor(
            run_mode=run_mode,
            configfile=self.cfg.getstr('processing_config_file'),
            test=self.args.test,
        )
        if self.rank == 2:
            self.processor.print_params()

    def convert_from_stream(self, frames):
        img_info = {
            "filename":'dummy_filename.img',
            "full_path":"dummy_path/dummy_filename.img",
            "state": "import",
            "proc_name": self.name,
            "proc_url": "tcp://{}:{}".format(self.cfg.getstr('host'), self.cfg.getstr(
                'port')),
            "series": -1,
            "frame": -1,
            "run_mode": None,
            "mapping": "",
            'exposure_time': 0.1,
        }
        return_frames = None

        if len(frames) <= 2:  # catch stand-alone header frame or end-of-series frame
            fdict = utils.decode_header(frames[0])
            try:
                assert "htype" in fdict
            except AssertionError:
                img_info["state"] = "error"
                img_info["dat_error"] = 'DATA ERROR: Invalid entry: no "hdict" key!'
            else:
                if "dseries_end" in fdict["htype"]:
                    img_info["state"] = "series-end"
                elif self.cfg.getstr('header_type') in fdict["htype"]:
                    try:
                        self.make_header(frames)
                    except Exception as e:
                        img_info["state"] = "error"
                        img_info["dat_error"] = "HEADER ERROR: {}".format(str(e))
                    else:
                        img_info["series"] = -999
                        img_info["frame"] = -999
                        img_info["state"] = "header-frame"
        else:
            hdr_frames = frames[:2]
            img_frames = frames[2:]
            self.make_header(frames=hdr_frames)
            try:
                # Get custom keys (if any) from header
                hdict = utils.decode_header(header=self.header[0])
                custom_keys_string = self.cfg.getstr('custom_keys')
                if custom_keys_string is not None:
                    custom_keys = [k.strip() for k in custom_keys_string.split(',')]
                    for ckey in custom_keys:
                        if ckey == self.cfg.getstr('filepath_key'):
                            img_info['filename'] = os.path.basename(hdict[ckey])
                            img_info['full_path'] = hdict[ckey]
                        else:
                            if self.cfg.getstr('run_mode_key') in ckey:
                                p_idx = self.cfg.getint('run_mode_key_index')
                                img_info["run_mode"] = hdict[ckey].split('.')[p_idx]
                            img_info[ckey] = hdict[ckey]

                # Get exposure time (frame time) from header
                hdict_1 = utils.decode_frame(frame=self.header[1])
                img_info['exposure_time'] = float(hdict_1['frame_time'])

                # Get frame info from frame
                fdict = utils.decode_frame_header(img_frames[0][:-1])
                img_info.update(
                    {"series": fdict["series"], "frame": fdict["frame"], }
                )
                img_info["state"] = "process"
                return_frames = img_frames
            except Exception as e:
                img_info["state"] = "error"
                img_info["dat_error"] = "CONVERSION ERROR: {}".format(str(e))
        return img_info, return_frames

    def make_header(self, frames):
        if isinstance(frames[0], bytes):
            self.header = [frames[0][:-1], frames[1][:-1]]
        else:
            self.header = [frames[0].bytes[:-1], frames[1].bytes[:-1]]

    def make_data_dict(self, frames):
        info, frames = self.convert_from_stream(frames)
        if info["state"] in ["error", "header-frame", "series-end"]:
            data = None
        else:
            data = {"header1": self.header[0], "header2": self.header[1]}
            for frm in frames:
                i = frames.index(frm) + 1
                key = "streamfile_{}".format(i)
                if i != 3:
                    data[key] = frm[:-1] if isinstance(frm, bytes) else frm.bytes[:-1]
                else:
                    data[key] = frm if isinstance(frm, bytes) else frm.bytes

        proc_info = {
            "beamXY": (0, 0),
            "dist": 0,
            "n_spots": 0,
            "n_overloads": 0,
            "hres": 99.0,
            "score": 0,
            "n_ice_rings": 0,
            "mean_shape_ratio": 0,
            "n_indexed": 0,
            "n_integrated": -999,
            "sg": "NA",
            "uc": "NA",
            "comment": "",
            "t0": 0,
            "phil": "",
        }
        info.update(proc_info)

        return data, info

    def process(self, info, frame, filename):
        s_proc = time.time()
        # regenerate processor if necessary
        if info['run_mode'] != self.processor.run_mode:
            self.generate_processor(run_mode=info['run_mode'])

        # process image
        info = self.processor.run(data=frame, filename=filename, info=info)
        info["proc_time"] = time.time() - s_proc
        return info

    def write_eiger_file(self):
        process_idx = self.rank
        detector = self.cfg.getstr('detector')
        beamline = self.cfg.getstr('beamline')
        filename = "data_{}_{}.stream".format(beamline, process_idx)
        with open(filename, "w") as fh:
            if detector is None:
                fh.write("DATASTREAM")
            elif "EIGER" in detector.upper():
                fh.write("EIGERSTREAM")
            elif "PILATUS" in detector.upper():
                fh.write("PILATUSSTREAM")
        return filename

    def initialize_zmq_sockets(self, init_r_socket=True):
        try:
            # If the Connector is active, connect the Reader socket to the Connector;
            # if not, connect the Reader socket to the Splitter
            if self.args.broker:
                dhost = self.localhost
                dport = "6{}".format(str(self.cfg.getstr('port'))[1:])
            else:
                dhost = self.cfg.getstr('host')
                dport = self.cfg.getstr('port')
            self.d_socket = self.make_socket(
                socket_type="req",
                wid=self.name,
                host=dhost,
                port=dport,
                verbose=self.args.verbose,
            )
            proc_url = "tcp://{}:{}".format(dhost, dport)

            if init_r_socket:
                # make r_socket either collector socket or UI socket
                if self.args.collect_results:
                    if self.args.collector_host:
                        chost = self.args.collector_host
                    else:
                        chost = self.localhost
                    cport = "7{}".format(str(self.cfg.getstr('port'))[1:])
                    self.r_socket = self.make_socket(
                        socket_type="push",
                        wid="{}_2C".format(self.name),
                        host=chost,
                        port=cport,
                        verbose=self.args.verbose,
                    )
                else:
                    self.r_socket = self.make_socket(
                        socket_type="push",
                        wid=self.name + "_2UIC",
                        host=self.cfg.getstr('uihost'),
                        port=self.cfg.getstr('uiport'),
                        verbose=True
                    )
                    self.r_socket.setsockopt(zmq.SNDTIMEO, 1000)
        except Exception as e:
            print("SOCKET ERROR: {}".format(e))
            exit()
        else:
            info = {
                "proc_name": self.name,
                "proc_url": proc_url,
                "state": "connected",
            }
            self.r_socket.send_json(info)

    def read_stream(self):
        # Write eiger_*.stream file
        filename = self.write_eiger_file()

        # Start listening for ZMQ stream
        while True:
            time_info = {
                "receive_time": 0,
                "wait_time": 0,
                "total_time": 0,
            }
            try:
                start = time.time()
                self.d_socket.send(self.name.encode('utf-8'))
                fstart = time.time()
                frames = self.d_socket.recv_multipart()
                time_info["receive_time"] = time.time() - fstart
                time_info["wait_time"] = time.time() - start - time_info[
                    "receive_time"]
                if self.args.broker:  # if it came from broker, remove first two frames
                    frames = frames[2:]

            except Exception as exp:
                print("DEBUG: {} CONNECT FAILED! {}".format(self.name, exp))
                continue
            else:
                # Drain images without processing
                if self.args.drain:
                    if self.args.verbose:
                        print(
                            str(frames[0][:-1])[3:-2],
                            "({})".format(self.name),
                            "rcv time: {:.4f} sec".format(time_info["receive_time"]),
                        )
                        if not hasattr(self, "recv_times"):
                            self.recv_times = [time_info["receive_time"]]
                        else:
                            self.recv_times.append(time_info["receive_time"])
                        print ('median receive time = ', np.median(self.recv_times))
                        time.sleep(0.01)
                else:
                    # make data and info dictionaries
                    data, info = self.make_data_dict(frames)

                    # handle different info scenarios
                    # some unknown error
                    if info is None:
                        continue
                    # normal processing info
                    elif info["state"] == "process":
                        info = self.process(info, frame=data, filename=filename)
                        time_info["total_time"] = time.time() - start
                        info.update(time_info)
                    # end-of-series signal (sleep for four seconds... maybe obsolete)
                    elif info["state"] == "series-end":
                        time.sleep(4)
                        continue

                    # send info to collector or direct to UI
                    if self.args.result_format == 'json':
                        self.r_socket.send_json(info)
                    elif self.args.result_format == 'string':
                        ui_msg = make_result_string(info=info, cfg=self.cfg)
                        self.r_socket.send_string(ui_msg)

        self.d_socket.close()

    def run(self):
        self.read_stream()

    def abort(self):
        self.stop = True


class Collector(ZMQProcessBase):
    """ Runs as 0-ranked process in MPI; collects all processing results from
      individual Reader processes and prints them to stdout and sends them
      off as a single stream to the UI if requested.
  """

    def __init__(self, name="COLLECTOR", comm=None, args=None, localhost=None):
        super(Collector, self).__init__(
            name=name, comm=comm, args=args, localhost=localhost
        )
        self.readers = {}
        self.advance_stdout = False

        # Debug
        self.proc_times = []
        self.recv_times = []

    def monitor_splitter_messages(self):
        # listen for messages from the splitter monitor port
        # todo: it occurs to me that this can be used for a variety of purposes!
        mport = "5{}".format(str(self.cfg.getstr('port'))[1:])
        self.m_socket = self.make_socket(
            socket_type="sub",
            wid=self.name + "_M",
            host=self.cfg.getstr('host'),
            port=mport,
            bind=True,
            verbose=True,
        )
        self.m_socket.setsockopt(zmq.SUBSCRIBE, b'')
        while True:
            msg = self.m_socket.recv()
            if msg:
                print('\n*** RUN FINISHED! ***\n')
                print(time.strftime('%b %d %Y %I:%M:%S %p'))
                msg_dict = utils.decode_frame(msg, tags='requests')
                checked_in = msg_dict['requests']
                hung_readers = []
                down_readers = []
                for rdr in self.readers.keys():
                    if not rdr in checked_in.keys():
                        down_readers.append(rdr)
                    elif not "series_end" in checked_in[rdr]:
                        hung_readers.append(rdr)
                if hung_readers:
                    print('{} Readers down during this run:'.format(len(hung_readers)))
                    for rdr in hung_readers:
                        lt = time.localtime(self.readers[rdr]['last_reported'])
                        silent_since = time.strftime('%b %d %Y %I:%M:%S %p', lt)
                        print('  {} silent since {}'.format(rdr, silent_since))
                if down_readers:
                    print('{} Readers are permanently down:'.format(len(down_readers)))
                    for rdr in down_readers:
                        lt = time.localtime(self.readers[rdr]['last_reported'])
                        silent_since = time.strftime('%b %d %Y %I:%M:%S %p', lt)
                        print('  {} down since {}'.format(rdr, silent_since))
                idle_readers = len(self.readers) - len(hung_readers) - len(down_readers)
                print('{} of {} Readers are CONNECTED and IDLE'.format(
                    idle_readers,
                    len(self.readers), ),
                    flush=True)
                self.send_check_in_info(hung_readers, down_readers)
                self.advance_stdout = True

    def send_check_in_info(self, hung_readers, down_readers):
        down_readers.extend(hung_readers)
        down_dict = {"down_readers": down_readers}
        self.broadcast(data=down_dict)

    def understand_info(self, info):
        reader_name = info['proc_name']
        if info["state"] == "connected":
            # add reader index to dictionary of active readers with state "ON"
            self.readers[reader_name] = {
                'name': reader_name,
                'status': 'IDLE',
                'start_time': time.time(),
            }
            msg = "{} CONNECTED to {}".format(reader_name, info["proc_url"])
            if len(self.readers) == self.size - 1:
                msg = '{} Readers connected ({})'.format(
                    len(self.readers),
                    time.strftime('%b %d %Y %I:%M:%S %p'),
                )
                self.advance_stdout = True
        elif info["state"] == "series-end":
            # change reader index in dictionary of active readers to "EOS"
            self.readers[reader_name]['status'] = 'IDLE'
            msg = "{} received END-OF-SERIES signal".format(reader_name)
        else:
            if reader_name in self.readers:
                self.readers[reader_name]['status'] = 'WORKING'
                if info["state"] == "error":
                    msg = "{} DATA ERROR: {}".format(info["proc_name"], info["dat_error"])
                elif info["state"] != "process":
                    msg = "DEBUG: {} STATE IS ".format(info["state"])
                else:
                    return False
            else:
                return False
        if self.args.verbose:
            print(msg, flush=True)
        return True

    def write_to_file(self, rlines):
        with open(self.args.record, "a") as rf:
            for rline in rlines:
                rf.write(rline)

    def print_to_stdout(self, counter, info, ui_msg):
        try:
            lines = [
                "*** [{}] ({}) SERIES {}, FRAME {} ({}):".format(
                    counter, info["proc_name"], info["series"], info["frame"],
                    info["full_path"]
                ),
                "  {}".format(ui_msg),
                "  TIME: wait = {:.4f} sec, recv = {:.4f} sec, "
                "proc = {:.4f} ,total = {:.2f} sec".format(
                    info["wait_time"],
                    info["receive_time"],
                    info["proc_time"],
                    info["total_time"],
                ),
                "***\n",
                "TEST INTEGRATION: {} INTEGRATED!".format(info['n_integrated']),
            ]
            self.proc_times.append(info['proc_time'])
            self.recv_times.append(info['receive_time'])
        except Exception as e:
            print(e)
        for ln in lines:
            print(ln)
            print("proc = {:.2f}, recv = {:.2f}".format(
                    np.median(self.proc_times), np.median(self.recv_times)))
        if self.args.record:
            self.write_to_file(lines)

    def initialize_zmq_sockets(self):
        cport = "7{}".format(str(self.cfg.getstr('port'))[1:])
        self.c_socket = self.make_socket(
            socket_type="pull",
            wid=self.name,
            host=self.localhost,
            port=cport,
            bind=True,
        )
        if self.cfg.getboolean('send_to_ui') or (self.cfg.getstr('uihost') and
                                                 self.cfg.getstr('uiport')):
            self.ui_socket = self.make_socket(
                socket_type="push",
                wid=self.name + "_2UI",
                host=self.cfg.getstr('uihost'),
                port=self.cfg.getstr('uiport'),
                verbose=True
            )
            self.ui_socket.setsockopt(zmq.SNDTIMEO, 1000)

    def collect_results(self):
        self.initialize_zmq_sockets()
        counter = 0
        while True:
            if self.c_socket.poll(timeout=500):
                # Accept information and (optionally) print to stdout
                if self.args.result_format == 'json':
                    info = self.c_socket.recv_json()
                    if info:
                        # understand info (if not regular info, don't send to UI)
                        if self.understand_info(info):
                            continue
                        else:
                            counter += 1
                        ui_msg = make_result_string(info=info, cfg=self.cfg)
                        if self.args.verbose:
                            self.print_to_stdout(counter=counter, info=info, ui_msg=ui_msg)
                        counter += 1
                elif self.args.result_format == 'string':
                    ui_msg = self.c_socket.recv_string()
                    if self.args.verbose:
                        print('[{:>8}] - {}'.format(counter, ui_msg))
                    counter += 1

                # send string to UI (DHS or Interceptor GUI)
                if self.cfg.getboolean('send_to_ui') or (self.cfg.getstr(
                        'uihost') and self.cfg.getstr('uiport')):
                    try:
                        self.ui_socket.send_string(ui_msg)
                    except Exception as e:
                        print('UI SEND ERROR: ', e)

            else:
                if self.advance_stdout:
                    self.advance_stdout = False
                    print('\n\n\n', flush=True)

    def run(self):
        report_thread = Thread(target=self.collect_results)
        monitor_thread = Thread(target=self.monitor_splitter_messages)
        report_thread.start()
        monitor_thread.start()

# -- end
