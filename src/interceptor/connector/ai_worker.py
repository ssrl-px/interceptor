from __future__ import division, absolute_import, print_function

import time

import numpy as np
import os
import zmq
from interceptor import packagefinder
from interceptor.connector import utils, make_result_string
from interceptor.connector.connector import ZMQProcessBase
from interceptor.connector.processor import InterceptorBaseProcessor, ZMQProcessor
from resonet.utils.predict_fabio import ImagePredictFabio


class AIProcessor(InterceptorBaseProcessor):
    def __init__(
            self,
            run_mode='file',
            configfile=None,
            test=False,
            verbose=False,
    ):
        self.verbose = verbose
        InterceptorBaseProcessor.__init__(self, run_mode=run_mode,
                                          configfile=configfile, test=test)

        # generate AI scorer
        self.scorer = AIScorer(config=self.cfg)

    def process(self, filename, info):

        info["n_spots"] = 0
        info["n_overloads"] = 0
        info["score"] = 0
        info["hres"] = -999
        info["n_ice_rings"] = 0
        info["mean_shape_ratio"] = 1
        info["sg"] = None
        info["uc"] = None

        try:
            load_start = time.time()
            # TODO: CHANGE TO ACTUAL HEADER READING
            self.scorer.predictor.load_image_from_file_or_array(
                image_file=filename,
                detdist=250,
                pixsize=0.075,
                wavelen=0.979,
            )
            print ("LOAD TIME: {}".format(time.time() - load_start))

            start = time.time()
            score = self.scorer.calculate_score()
            print ('SCORING TIME: {}'.format(time.time() - start))
            info['score'] = score
        except Exception as err:
            import traceback
            traceback.print_exc()
            spf_tb = traceback.format_exc()
            info["spf_error"] = "SPF ERROR: {}".format(str(err))
            info['spf_error_tback'] = spf_tb
            return info
        else:
            info['n_spots'] = self.scorer.n_spots
            info["hres"] = self.scorer.hres
            info["n_ice_rings"] = self.scorer.n_ice_rings
            info["split"] = self.scorer.split
            info['spf_error'] = 'SPLITTING = {}'.format(self.scorer.split)
        return info

    def run(self, filename, info):
        start = time.time()
        info = self.process(filename, info)
        info['proc_time'] = time.time() - start
        return info


class AIScorer(object):
    """
    A wrapper class for the neural network image evaluator(s), which will hopefully find a) resolution,
    b) split spots / multiple lattices, c) scattering rings
    """
    def __init__(self, config):
        self.cfg = config
        self.hres = -999
        self.n_ice_rings = 0
        self.split = False
        self.n_spots = 0

        # Check for custom models and architectures
        reso_model = self.cfg.getstr('resolution_model')
        reso_arch = self.cfg.getstr('resolution_architecture')
        multi_model = self.cfg.getstr('multilattice_model')
        multi_arch = self.cfg.getstr('multilattice_architecture')
        spf_model = self.cfg.getstr('spotfinding_model')
        spf_arch = self.cfg.getstr('spotfinding_architecture')
        use_b2d = self.cfg.getstr('use_b_to_d')
        b2d_model = self.cfg.getstr('b_to_d_model')


        # Generate predictor
        assert reso_model is not None
        assert multi_model is not None
        assert spf_arch is not None
        self.predictor = ImagePredictFabio(
            reso_model=reso_model,
            reso_arch=reso_arch,
            multi_model=multi_model,
            multi_arch=multi_arch,
            ice_model=None,
            ice_arch=None,
            counts_model=spf_model,
            counts_arch=spf_arch,
        )

    def count_spots(self):
        return self.predictor.count_spots()

    def estimate_resolution(self):
        res = self.predictor.detect_resolution()
        if np.isnan(res):
            res = 99.0
        return res

    def find_rings(self):
        return 0

    def find_multilattice(self):
        return self.predictor.detect_multilattice_scattering(binary=self.cfg.getboolean('multilattice_binary'))*100

    def calculate_score(self):
        score = 0
        self.hres = self.estimate_resolution()
        res_score = [
            (20, 1),
            (8, 2),
            (5, 3),
            (4, 4),
            (3.2, 5),
            (2.7, 7),
            (2.4, 8),
            (2.0, 10),
            (1.7, 12),
            (1.5, 14),
        ]
        if self.hres > 20:
            score -= 2
        else:
            increment = 0
            for res, inc in res_score:
                if self.hres < res:
                    increment = inc
            score += increment

        # evaluate ice ring presence
        self.n_ice_rings = self.find_rings()
        if self.n_ice_rings >= 4:
            score -= 3
        elif 4 > self.n_ice_rings >= 2:
            score -= 2
        elif self.n_ice_rings == 1:
            score -= 1

        # evaluate splitting
        self.split = self.find_multilattice()

        # count spots
        self.n_spots = self.count_spots()

        return score


class AIWorker(ZMQProcessBase):
    """ ZMQ AIWorker: requests a single frame (multipart) from the Eiger,
      converts to dictionary format, attaches to a special format class,
      carries out processing, and sends the result via ZMQ connection to the
      Collector class.
  """

    def __init__(self, name="zmq_reader", comm=None, args=None, localhost="localhost"):
        super(AIWorker, self).__init__(
            name=name, comm=comm, args=args, localhost=localhost
        )
        self.name = "ZMQ_{:03d}".format(self.rank)
        self.detector = self.find_detector()
        self.generate_processor()

        # Initialize ZMQ sockets
        self.initialize_zmq_sockets()

    def find_detector(self):
        import json
        try:
            det_file = self.cfg.getstr('detector_registry_file')
            if det_file is None:
                det_file = packagefinder('detector.json', 'format', read_config=False)
            with open(det_file, 'r') as jf:
                detector_dict = json.load(jf)
            det_key = self.cfg.getstr('detector').upper()
            detector = detector_dict[det_key]
            return detector
        except Exception as e:
            print("WARNING: DETECTOR NOT FOUND! {}".format(str(e)))
            return None

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
            "split": 0,
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

    def process(self, info, frame):
        s_proc = time.time()
        # regenerate processor if necessary
        if info['run_mode'] != self.processor.run_mode:
            self.generate_processor(run_mode=info['run_mode'])

        # process image
        info = self.processor.run(data=frame, info=info, detector=self.detector)
        info["proc_time"] = time.time() - s_proc
        return info

    def initialize_zmq_sockets(self, init_r_socket=True):
        try:
            # If the Connector is active, connect the AIWorker socket to the Connector;
            # if not, connect the AIWorker socket to the Splitter
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
                        info = self.process(info, frame=data)
                        time_info["total_time"] = time.time() - start
                        info.update(time_info)
                    # end-of-series signal (sleep for four seconds... maybe obsolete)
                    elif info["state"] == "series-end":
                        time.sleep(self.detector['timeout'])
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
