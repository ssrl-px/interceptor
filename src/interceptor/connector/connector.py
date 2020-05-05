from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 03/31/2020
Description : Streaming stills processor for live data analysis
"""

import os
import time

import zmq

from dxtbx.model.experiment_list import ExperimentListFactory

from iota.components.iota_init import initialize_single_image
from interceptor.connector.processor import FastProcessor, IOTAProcessor
from interceptor.connector.stream import ZMQStream
from interceptor.format import FormatEigerStreamSSRL as FormatStream


class ConnectorBase:
    """ Base class for ZMQReader and ZMQCollector classes """

    def __init__(self, comm, args, name="zmq_thread", localhost="localhost"):
        """ Constructor
    :param comm: mpi4py communication instance
    :param args: command line arguments
    :param name: thread name (for logging mostly)
    """
        self.name = name
        self.comm = comm
        self.localhost = localhost

        if comm:
            self.rank = comm.Get_rank()  # each process in MPI has a unique id
            self.size = comm.Get_size()  # number of processes running in this job
        else:
            self.rank = 0
            self.size = 0

        self.stop = False
        self.timeout_start = None
        self.args = args

    def generate_processor(self, args):
        if args.iota:
            dummy_path = os.path.abspath(os.path.join(os.curdir, "dummy_file.h5"))
            info, iparams = initialize_single_image(img=dummy_path, paramfile=None)
            processor = IOTAProcessor(info, iparams, last_stage=args.last_stage)
        else:
            processor = FastProcessor(
                last_stage=args.last_stage, test=args.test, phil_file=args.phil
            )
        return processor

    def initialize_process(self):
        if self.comm:
            # Generate params, args, etc. if process rank id = 0
            if self.rank == 0:
                processor = self.generate_processor(self.args)
                info = dict(
                    processor=processor,
                    args=self.args,
                    host=self.args.host,
                    port=self.args.port,
                    rhost=self.localhost,
                    rport="7{}".format(str(self.args.port)[1:]),
                )
            else:
                info = None

            # send info dict to all processes
            info = self.comm.bcast(info, root=0)

            # extract info
            self.processor = info["processor"]
            self.args = info["args"]
            self.host = info["host"]
            self.port = info["port"]
            self.rhost = info["rhost"]
            self.rport = info["rport"]
        else:
            self.processor = self.generate_processor(self.args)
            self.host = self.args.host
            self.port = self.args.port

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
                    print("\n****** TIMED OUT! ******")
                    exit()


class Reader(ConnectorBase):
    """ ZMQ Reader: requests a single frame (multipart) from the Eiger,
      converts to dictionary format, attaches to a special format class,
      carries out processing, and sends the result via ZMQ connection to the
      Collector class.
  """

    def __init__(self, name="zmq_reader", comm=None, args=None):
        super(Reader, self).__init__(name=name, comm=comm, args=args)
        self.initialize_process()

        if self.rank == 1:
            self.processor.print_params()

    def convert_from_stream(self, frames):
        img_info = {
            "state": "process",
            "proc_name": self.name,
            "proc_url": "tcp://{}:{}".format(self.host, self.port),
            "run_no": -1,
            "frame_idx": -1,
            "mapping": "",
            "reporting": "",
            "filename": "",
        }
        return_frames = None

        if len(frames) < 2:  # catch end frame
            framestring = frames[0] if isinstance(frames[0], bytes) else frames[0].bytes
            if "dseries-end" in framestring:
                img_info["state"] = "series-end"
        elif len(frames) == 2:  # catch header frame (if applicable)
            framestring = frames[0] if isinstance(frames[0], bytes) else frames[0].bytes
            if self.args.header and self.args.htype in str(framestring):
                self.make_header(frames)
                img_info["run_no"] = -999
                img_info["frame_idx"] = -999
                img_info["state"] = "header-frame"
        else:
            try:
                if not self.args.header:
                    hdr_frames = frames[:2]
                    img_frames = frames[2:]
                    self.make_header(frames=hdr_frames)
                else:
                    img_frames = frames
            except Exception as e:
                img_info["state"] = "error"
                img_info["dat_error"] = "HEADER ERROR: {}".format(str(e))
            else:
                try:
                    # Get master_file name from header
                    header_split = str(self.header[0][3:-2]).split(",")
                    for part in header_split:
                        if "master_file" in part:
                            filepath = part.split(":")[1].strip('"')
                            img_info["filename"] = os.path.basename(filepath)
                        if "mapping" in part:
                            img_info["mapping"] = part.split(":")[1].strip('"')
                        if "reporting" in part:
                            img_info["reporting"] = part.split(":")[1].strip('"')

                    # Get frame info from frame
                    if isinstance(img_frames[0], bytes):
                        frame_string = str(img_frames[0][:-1])[3:-2]
                    else:
                        frame_string = str(img_frames[0].bytes[:-1])[3:-2]
                    frame_split = frame_string.split(",")
                    for part in frame_split:
                        if "series" in part:
                            img_info["run_no"] = part.split(":")[1]
                        if "frame" in part:
                            img_info["frame_idx"] = part.split(":")[1]
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
        info = self.processor.run(data=frame, filename=filename, info=info)
        info["proc_time"] = time.time() - s_proc
        return info

    def write_eiger_file(self):
        eiger_idx = self.rank
        filename = "eiger_{}.stream".format(eiger_idx)
        self.name = "ZMQ_{:03d}".format(eiger_idx)
        with open(filename, "w") as fh:
            fh.write("EIGERSTREAM")
        return filename

    def initialize_zmq_sockets(self):
        # Initialize ZMQ stream listener (aka 'data socket')
        try:
            self.d_socket = ZMQStream(
                name=self.name,
                host=self.host,
                port=self.port,
                socket_type=self.args.stype,
            )

            # intra-process communication (aka 'result socket')
            self.r_socket = ZMQStream(
                name="{}_2C".format(self.name),
                host=self.rhost,
                port=self.rport,
                socket_type="push",
            )
        except Exception as e:
            print("SOCKET ERROR: {}".format(e))
            exit()
        else:
            info = {
                "proc_name": self.name,
                "proc_url": "tcp://{}:{}".format(self.host, self.port),
                "state": "connected",
            }
            self.r_socket.send_json(info)

    def process_stream(self):
        # Write eiger_*.stream file
        filename = self.write_eiger_file()

        # Initialize ZMQ sockets
        self.initialize_zmq_sockets()

        # Start listening for ZMQ stream
        while True:
            start = time.time()
            try:
                fstart = time.time()
                if self.args.stype.lower() == "req":
                    self.d_socket.send(b"Hello")
                    expecting_reply = True
                    while expecting_reply:
                        if self.d_socket.poll(timeout=10000):
                            frames = self.d_socket.receive(copy=False, flags=0)
                            expecting_reply = False
                        else:
                            self.d_socket.reset()
                            self.d_socket.send(b"Hello")
                else:
                    frames = self.d_socket.receive(copy=False, flags=0)
                fel = time.time() - fstart
            except Exception as exp:
                print("DEBUG: {} CONNECT FAILED! {}".format(self.name, exp))
                continue
            else:
                # Drain images without processing
                if self.args.drain:
                    if self.args.verbose:
                        print(
                            str(frames[2].bytes[:-1])[3:-2],
                            "({})".format(self.name),
                            "rcv time: {:.4f} sec".format(fel),
                        )
                    continue

                data, info = self.make_data_dict(frames)

                if info is None:
                    print("debug: info is None!")
                    continue
                elif info['state'] == 'process':
                    info = self.process(info, frame=data, filename=filename)
                    elapsed = time.time() - start
                    time_info = {"total_time": elapsed, "receive_time": fel}
                    info.update(time_info)

                # send info to collector
                self.r_socket.send_json(info)

            finally:
                # stop if called
                if self.stop:
                    break

        self.d_socket.close()

    def run(self):
        self.process_stream()

    def abort(self):
        self.stop = True


class Collector(ConnectorBase):
    """ Runs as 0-ranked process in MPI; collects all processing results from
      individual Reader processes and prints them to stdout and sends them
      off as a single stream to the UI if requested.
  """

    def __init__(self, name="ZMQ_000", comm=None, args=None, localhost=None):
        super(Collector, self).__init__(
            name=name, comm=comm, args=args, localhost=localhost
        )
        self.initialize_process()

    def understand_info(self, info):
        if info["state"] == "connected":
            msg = "{} CONNECTED to {}".format(info["proc_name"], info["proc_url"])
        elif info["state"] == "series-end":
            msg = "{} received END-OF-SERIES signal".format(info["proc_name"])
        elif info["state"] == "error":
            msg = '{} DATA ERROR: {}'.format(info["proc_name"], info["dat_error"])
        else:
            return False
        if self.args.verbose:
            print(msg)
        return True

    def write_to_file(self, info):
        with open(self.args.record, "a") as rf:
            rline = "{:<8} {:<4} {:<10.8f} {}\n".format(
                info["proc_name"], info["frame_idx"], info["t0"], info["proc_time"],
            )
            rf.write(rline)

    def make_result_string(self, info):
        # message to DHS / UI
        err_list = [
            e for e in info if ("error" in e or "comment" in e) and info[e] != ""
        ]
        errors = "; ".join(err_list)
        results = (
            "{0} {1} {2} {3:.2f} {4} "
            "{5:.2f} {6} {7} {{{8}}}"
            "".format(
                info["n_spots"],  # number_of_spots
                info["n_overloads"],  # number_of_spots_with_overloaded_pixels
                info["score"],  # composite score (used to be n_indexed...)
                info["hres"],  # high resolution boundary
                info["n_ice_rings"],  # number_of_ice-rings
                info["mean_shape_ratio"],  # mean spot shape ratio
                info["sg"],  # space group
                info["uc"],  # unit cell
                errors,  # errors
            )
        )
        reporting = (
            info["reporting"] if info["reporting"] != "" else "htos_log note zmqDhs"
        )
        ui_msg = (
            "{0} run {1} frame {2} result {{{3}}} mapping {{{4}}} "
            "filename {5}".format(
                reporting,
                info["run_no"],  # run number
                info["frame_idx"],  # frame index
                results,  # processing results
                info["mapping"],  # mapping from run header
                info["filename"],  # master file
            )
        )
        return ui_msg

    def print_to_stdout(self, info, ui_msg):
        print(
            "*** ({}) RUN {}, FRAME {}:".format(
                info["proc_name"], info["run_no"], info["frame_idx"]
            )
        )
        print("  {}".format(ui_msg))
        print(
            "  TIME: recv = {:.2f} sec,"
            " proc = {:.4f} ,"
            " total = {:.2f} sec".format(
                info["receive_time"], info["proc_time"], info["total_time"],
            ),
        )
        print("***\n")

    def output_results(self, info, record=False, verbose=False):
        ui_msg = None
        if record:
            self.write_to_file(info=info)
        try:
            ui_msg = self.make_result_string(info=info)
        except Exception as exp:
            print("PRINT ERROR: ", exp)
        else:
            if verbose:
                self.print_to_stdout(info=info, ui_msg=ui_msg)
        finally:
            return ui_msg

    def collect(self):
        collector = ZMQStream(
            name=self.name,
            host=self.rhost,
            port=self.rport,
            socket_type="pull",
            bind=True,
        )

        send_to_ui = self.args.send or (self.args.uihost and self.args.uiport)
        if send_to_ui:
            ui_socket = ZMQStream(
                name=self.name + "_2C",
                host=self.args.uihost,
                port=self.args.uiport,
                socket_type=self.args.uistype,
            )
        while True:
            info = collector.receive_json()
            if info:
                # understand info (if not regular info, don't send to UI)
                if self.understand_info(info):
                    continue

                # send string to UI (DHS or Interceptor GUI)
                ui_msg = self.output_results(
                    info, record=self.args.record, verbose=self.args.verbose
                )
                if send_to_ui:
                    try:
                        ui_socket.send_string(ui_msg)
                    except Exception:
                        pass

    def run(self):
        self.collect()


# -- end
