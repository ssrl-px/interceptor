from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 05/15/2020
Description : Streaming stills processor for live data analysis
"""

import time
import zmq

import numpy as np

from threading import Thread

from interceptor import packagefinder, read_config_file
from interceptor.connector import utils
from interceptor.connector import make_result_string


def debug_segfault():
    """ Deliberate segfault for debugging purposes """
    import ctypes

    ctypes.string_at(1)


class ZMQProcessBase:
    """ Base class for Connector, AIWorker, and Collector classes """

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
    from Splitter and assign each image to the least-recently used (LRU) AIWorker,
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


class Collector(ZMQProcessBase):
    """ Runs as 0-ranked process in MPI; collects all processing results from
      individual AIWorker processes and prints them to stdout and sends them
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

    def csv_from_json(self, info):
        # collect errors
        err_list = [
            info[e] for e in info if ("error" in e or "comment" in e) and info[e] != ""
        ]
        errors = "; ".join(err_list)
        line = "{0},{1},{2},{3:.2f},{4},{5:.2f},{6},{7},{8},{9},{10},{11}".format(
            info["series"],
            info["frame"],
            info["full_path"],
            info["n_spots"],  # number_of_spots
            info["n_overloads"],  # number_of_spots_with_overloaded_pixels
            info["score"],  # composite score (used to be n_indexed...)
            info["hres"],  # high resolution boundary
            info["n_ice_rings"],  # number_of_ice-rings
            info["mean_shape_ratio"],  # mean spot shape ratio
            info["sg"],  # space group
            info["uc"],  # unit cell
            errors,  # errors
            '\n',
            )
        return line

    def collect_results(self):
        self.initialize_zmq_sockets()
        counter = 0
        while True:
            if self.c_socket.poll(timeout=500):
                # Accept information and (optionally) print to stdout
                if self.args.result_format == 'json':
                    info = self.c_socket.recv_json()
                    if self.args.record:
                        if counter == 0:
                            if self.rank == 2:
                                print ("COLUMN NAMES GO HERE!")
                        else:
                            line = self.csv_from_json(info=info)
                            self.write_to_file([line])
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
