from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 05/15/2020
Description : Streaming stills processor for live data analysis
"""
import time  # noqa: F401; keep around for testing
import os

import numpy as np
import importlib
import json

from cctbx import sgtbx, crystal
from iotbx import phil as ip

from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.imageset import ImageSet, ImageSetData, MemReader


from interceptor import packagefinder, read_config_file

from interceptor.format import extract_data

#AI Stuff
from resonet.utils.predict_fabio import ImagePredictFabio


class MemReaderNamedPath(MemReader):

  def __init__(self, path,  *args, **kwargs):
    self.dummie_path_name = path
    super(MemReaderNamedPath, self).__init__(*args, **kwargs)


class AIScorer(object):
    """
    A wrapper class for the neural network image evaluator(s), which will hopefully find a) resolution,
    b) split spots / multiple lattices, c) scattering rings
    """
    def __init__(self, config, dev="cpu"):
        self.cfg = config
        self.hres = -999
        self.n_ice_rings = 0
        self.split = False
        self.n_spots = 0

        self.dev=dev
        print(f"Using {dev} for processing!")

        # Check for custom models and architectures
        reso_model = self.cfg.getstr('resolution_model')
        reso_arch = self.cfg.getstr('resolution_architecture')
        multi_model = self.cfg.getstr('multilattice_model')
        multi_arch = self.cfg.getstr('multilattice_architecture')
        counts_model = self.cfg.getstr('spotfinding_model')
        counts_arch = self.cfg.getstr('spotfinding_architecture')
        if self.cfg.getboolean('use_b_to_d'):
            b2d_model = self.cfg.getstr('b_to_d_model')
        else:
            b2d_model = None

        # Generate predictor
        assert reso_model is not None
        assert multi_model is not None
        self.predictor = ImagePredictFabio(
            reso_model=reso_model,
            reso_arch=reso_arch,
            multi_model=multi_model,
            multi_arch=multi_arch,
            counts_model=counts_model,
            counts_arch=counts_arch,
            ice_model=None,
            ice_arch=None,
            B_to_d=b2d_model,
            dev=self.dev,
        )

        #TODO: add to config
        self.predictor.quads = list(map(int, self.cfg.getstr('detector_quadrants').split(',')))

    def estimate_resolution(self):
        res = self.predictor.detect_resolution()
        if np.isnan(res):
            res = 99.0
        return res

    def find_rings(self):
        return 0

    def find_multilattice(self):
        return self.predictor.detect_multilattice_scattering(binary=self.cfg.getboolean('multilattice_binary'))*100

    def find_n_spots(self):
        return self.predictor.count_spots()

    def calculate_score(self):
        score = 0
        start = time.time()
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

        # count the bragg peaks using AI model
        self.n_spots = self.find_n_spots()
        return score



class InterceptorBaseProcessor(object):
    def __init__(self, run_mode='DEFAULT', configfile=None, test=False):
        self.test = test
        self.run_mode = run_mode

        # generate processing config params
        if configfile and configfile != "None":
            p_config = read_config_file(configfile)
        else:
            print ('*** WARNING! Processing config file not found, generating default parameters!')
            p_config = packagefinder('processing.cfg', 'connector', read_config=True)

        try:
            self.cfg = p_config[run_mode]
        except KeyError:
            self.cfg = p_config['DEFAULT']

        if run_mode != "DEFAULT":
            self.processing_mode = run_mode
        else:
            self.processing_mode = self.cfg.getstr('processing_mode')

    @staticmethod
    def make_experiments(filename=None, data=None, detector=None):
        # make experiments
        e_start = time.time()
        experiments = None
        if data is None:
            # Regular ExperimentList creation with regular files
            assert filename is not None
            experiments = ExperimentListFactory.from_filenames([filename])

        elif data is not None:
            # ExperimentList creation with ZeroMQ format classes
            assert detector is not None
            load_models = True
            fc_string = "interceptor.format.{}".format(detector['format_class'])
            fc_module = importlib.import_module(fc_string)
            fc_module.injected_data = data
            fc_class = getattr(fc_module, detector['format_class'])
            format_class = fc_class(image_file=filename)

            # Inject data and create imageset
            reader = MemReaderNamedPath("virtual_datastream_path", [format_class])
            reader.format_class = format_class
            imageset_data = ImageSetData(reader, None)
            imageset = ImageSet(imageset_data)
            imageset.set_beam(format_class.get_beam())
            imageset.set_detector(format_class.get_detector())

            # Create an ExperimentList object from imageset
            experiments = ExperimentListFactory.from_stills_and_crystal(imageset, crystal=None, load_models=load_models)

        e_time = time.time() - e_start
        return experiments, e_time



class ZMQProcessor(InterceptorBaseProcessor):
    def __init__(
            self,
            run_mode='DEFAULT',
            configfile=None,
            detectorfile=None,
            test=False,
            dev="cpu",
    ):
        InterceptorBaseProcessor.__init__(self, run_mode=run_mode,
                                          configfile=configfile,
                                          test=test)
        self.ai_scorer = AIScorer(config=self.cfg, dev=dev)

        self.sp_scorer = None # deprecated

    def process(self, data, detector, info):
        #info["phil"] = self.dials_phil.as_str()
        filename = info['full_path']

        # Make ExperimentList object
        experiments, e_time = self.make_experiments(data=data, detector=detector, filename=filename)

        try:
            if data is None:
                panel = experiments[0].detector[0]
                beam = experiments[0].beam
                try:
                    raw_data = experiments[0].imageset.get_raw_data()
                except:
                    raw_data = experiments[0].imageset.get_raw_data(0)
                if isinstance(raw_data, tuple):
                    raw_data = raw_data[0]
                raw_data = raw_data.as_numpy_array()
                if not raw_data.dtype==np.float32:
                    raw_data = raw_data.astype(np.float32)
                detdist = panel.get_distance()
                beam_x, beam_y = panel.get_beam_centre_px(beam.get_unit_s0())
                pixsize = panel.get_pixel_size()[0]
                wavelen = beam.get_wavelength()

            else:
                assert data is not None
                encoding_info = json.loads(data.get("streamfile_2", ""))
                raw_bytes = data.get("streamfile_3", "")
                header = json.loads(data.get("header2", ""))

                raw_data = extract_data(info=encoding_info, data=raw_bytes)
                if raw_data.dtype != np.float32:
                    raw_data = raw_data.astype(np.float32)
                beam_x = header["beam_center_x"]
                beam_y = header["beam_center_y"]
                detdist = header["detector_distance"]*1000
                wavelen = header["wavelength"]
                pixsize = header["x_pixel_size"]*1000

            self.ai_scorer.predictor.load_image_from_file_or_array(
                raw_image=raw_data,
                detdist=detdist,
                pixsize=pixsize,
                wavelen=wavelen,
                beam_center=(beam_x, beam_y),
                use_ice_mask=self.cfg.getboolean("resonet_mask_ice_rings")
            )
            score = self.ai_scorer.calculate_score()  # Once the score works, will replace
            # TODO add n_spots to info in calculate_score
        except Exception as err:
            import traceback
            traceback.print_exc()
            spf_tb = traceback.format_exc()
            info["spf_error"] = "SPF ERROR: {}".format(str(err))
            info['spf_error_tback'] = spf_tb
            return info
        else:
            info["hres"] = self.ai_scorer.hres
            info["split"] = self.ai_scorer.split
            info["n_spots"] = self.ai_scorer.n_spots

        # Doing it here because scoring can reject spots within ice rings, which can
        # drop the number below the minimal limit
        if info['n_spots'] < self.cfg.getint('min_Bragg_peaks'):
            info["spf_error"] = "Too few ({}) spots found!".format(
                info['n_spots'])

        # if last stage was selected to be "spotfinding", stop here; otherwise
        # perform a speed check before proceeding to indexing
        return info
        #if self.cfg.getstr('processing_mode') == "spotfinding" or info["n_spots"] <= \
        #        self.cfg.getint('min_Bragg_peaks'):
        #    return info
        #else:
        #    exposure_time_cutoff = self.cfg.getfloat('exposure_time_cutoff')
        #    if exposure_time_cutoff > info['exposure_time']:
        #        return info

    def run(self, data, detector, info):
        return self.process(data, detector, info)


# -- end

