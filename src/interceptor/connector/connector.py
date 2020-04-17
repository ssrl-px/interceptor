from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 03/31/2020
Description : Streaming stills processor for live data analysis
"""

import os
import time

from dxtbx.model.experiment_list import ExperimentListFactory

from iota.components.iota_init import initialize_single_image
from interceptor.connector.processor import FastProcessor, IOTAProcessor
from interceptor.connector.stream import ZMQStream
from interceptor.format import FormatEigerStreamSSRL as FormatStream


class ConnectorBase:
    """ Base class for ZMQReader and ZMQCollector classes """

    def __init__(self, comm, args, name="zmq_thread"):
        """ Constructor
    :param comm: mpi4py communication instance
    :param args: command line arguments
    :param name: thread name (for logging mostly)
    """
        self.name = name
        self.comm = comm

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
                # args, _ = self.generate_args()
                processor = self.generate_processor(self.args)
                info = dict(
                    processor=processor,
                    args=self.args,
                    host=self.args.host,
                    port=self.args.port,
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
        msg = ''
        if len(frames) <= 2:
            framestring = frames[0] if isinstance(frames[0], bytes) else frames[0].bytes
            if self.args.header and self.args.htype in str(framestring):
                self.make_header(frames)
                return [-999, -999, None], 'Header Frame'
        else:
            try:
                if not self.args.header:
                    hdr_frames = frames[:2]
                    img_frames = frames[2:]
                    if not hasattr(self, "header"):
                        self.make_header(frames=hdr_frames)
                else:
                    img_frames = frames
                if isinstance(img_frames[0], bytes):
                    frame_string = str(img_frames[0][:-1])[3:-2]  # extract dict entries
                else:
                    frame_string = str(img_frames[0].bytes[:-1])[3:-2]
                frame_split = frame_string.split(",")
                idx = -1
                run_no = -1
                for part in frame_split:
                    if "series" in part:
                        run_no = part.split(":")[1]
                    if "frame" in part:
                        idx = part.split(":")[1]
                return_frames = [run_no, idx, img_frames]
            except Exception as e:
                return_frames = None
                msg = 'CONVERSION ERROR: {}'.format(str(e))
            return return_frames, msg

    def make_header(self, frames):
        if isinstance(frames[0], bytes):
            self.header = [frames[0][:-1], frames[1][:-1]]
        else:
            self.header = [frames[0].bytes[:-1], frames[1].bytes[:-1]]

    def make_data_dict(self, frames):
        frames, msg = self.convert_from_stream(frames)
        if frames is None:
            return None, msg
        elif frames[0] == -999:
            return "header_frame", None
        else:
            run_no = frames[0]
            idx = frames[1]
            data = {"header1": self.header[0], "header2": self.header[1]}
            for frm in frames[2]:
                i = frames[2].index(frm) + 1
                key = "streamfile_{}".format(i)
                if i != 3:
                    data[key] = frm[:-1] if isinstance(frm, bytes) else frm.bytes[:-1]
                else:
                    data[key] = frm if isinstance(frm, bytes) else frm.bytes
            msg = ''

        info = {
            "proc_name": self.name,
            "run_no": run_no,
            "frame_idx": idx,
            "beamXY": (0, 0),
            "dist": 0,
            "n_spots": 0,
            "hres": 99.0,
            "n_indexed": 0,
            "sg": "NA",
            "uc": "NA",
            "dat_error": msg,
            "spf_error": "",
            "idx_error": "",
            "rix_error": "",
            "img_error": "",
            "prc_error": "",
            "comment": "",
            "t0": 0,
            "phil": "",
        }

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
        # Initialize ZMQ stream listener
        self.stream = ZMQStream(
            name=self.name, host=self.host, port=self.port, socket_type=self.args.stype
        )

        # intra-process communication (not using MPI... yet)
        self.uplink = ZMQStream(
            name="{}_uplink".format(self.name),
            host="localhost",
            port=7121,
            socket_type="push",
        )

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
                    self.stream.send(b"Hello")
                frames = self.stream.receive(copy=False, flags=0)
                fel = time.time() - fstart
            except Exception as exp:
                print("DEBUG: {} CONNECT FAILED! {}".format(self.name, exp))
                continue

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

            if data is 'header_frame':
                continue
            elif data is None:
                print (info)
                continue

            info = self.process(info, frame=data, filename=filename)
            elapsed = time.time() - start
            time_info = {"total_time": elapsed, "receive_time": fel}
            info.update(time_info)
            self.uplink.send_json(info)

            # If frame processing fits within specified interval, sleep for the
            # remainder of that interval; otherwise (or if args.interval == 0) don't
            # sleep at all
            interval = self.args.interval - elapsed
            if interval > 0:
                time.sleep(interval)

            if self.stop:
                break
        self.stream.close()

    def run(self):
        self.process_stream()

    def abort(self):
        self.stop = True


class Collector(ConnectorBase):
    """ Runs as 0-ranked process in MPI; collects all processing results from
      individual Reader processes and prints them to stdout and sends them
      off as a single stream to the UI if requested.
  """

    def __init__(self, name="ZMQ_000", comm=None, args=None):
        super(Collector, self).__init__(name=name, comm=comm, args=args)
        self.initialize_process()

    def run(self):
        # todo: decide whether to leave this port hardcoded
        host = "localhost"
        port = 7121
        socket_type = "pull"
        collector = ZMQStream(
            name=self.name, host=host, port=port, socket_type=socket_type, bind=True
        )

        send_to_ui = self.args.send or (self.args.uihost and self.args.uiport)
        if send_to_ui:
            ui_stream = ZMQStream(
                name=self.name + "_uplink",
                host=self.args.uihost,
                port=self.args.uiport,
                socket_type=self.args.uistype,
            )

        while True:
            info = collector.receive_json()
            if info:
                if self.args.record:
                    with open(self.args.record, "a") as rf:
                        rline = "{:<8} {:<4} {:<10.8f} {}\n".format(
                            info["proc_name"],
                            info["frame_idx"],
                            info["t0"],
                            info["prc_time"],
                        )
                        rf.write(rline)

                try:
                    # message to DHS / UI
                    prefix = "htos_log note zmaDhs"
                    err_list = [
                        info["dat_error"],
                        info["img_error"],
                        info["spf_error"],
                        info["idx_error"],
                        info["rix_error"],
                        info["prc_error"],
                        info["comment"],
                    ]
                    errors = ";".join([i for i in err_list if i != ""])
                    ui_msg = (
                        "{0} {1} {2} {3} {4} "
                        "{5} {6:.2f} {7} {8} {9} {{{10}}}"
                        "".format(
                            prefix,  # required for DHS logging
                            info["run_no"],  # run number
                            info["frame_idx"],  # frame index
                            info["n_spots"],  # number_of_spots
                            0,  # TODO: number_of_spots_with_overloaded_pixels
                            info["n_indexed"],  # number of indexed reflections
                            info["hres"],  # high resolution boundary
                            0,  # TODO: number_of_ice-rings
                            info["sg"],  # space group
                            info["uc"],  # unit cell
                            errors,  # errors
                        )
                    )
                except Exception as exp:
                    print("PRINT ERROR: ", exp)
                else:
                    if self.args.verbose:
                        print(
                            "*** ({}) RUN {}, FRAME {}:".format(
                                info["proc_name"], info["run_no"], info["frame_idx"]
                            )
                        )
                        print("  {}".format(ui_msg))
                        # print ("  DEBUG: BEAM X = {:.2f}, Y = {:.2f}, DIST = {:.2f}".format(
                        #   info['beamXY'][0], info['beamXY'][1], info['dist']
                        # ))
                        print(
                            "  TIME: recv = {:.2f} sec,"
                            " exp = {:.4f},"
                            " proc = {:.4f} ,"
                            " total = {:.2f} sec".format(
                                info["receive_time"],
                                info["exp_time"],
                                info["prc_time"],
                                info["total_time"],
                            ),
                        )
                        print("***\n")

                    # send string to UI (DHS and/or Interceptor GUI)
                    if send_to_ui:
                        try:
                            ui_stream.send_string(ui_msg)
                        except Exception as exp:
                            pass
