from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 05/15/2020
Description : Streaming stills processor for live data analysis
"""
import time  # noqa: F401; keep around for testing

import numpy as np
import importlib
import json

from cctbx import sgtbx, crystal
from iotbx import phil as ip

from dials.algorithms.spot_finding import per_image_analysis
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.imageset import ImageSet, ImageSetData, MemReader

from cctbx import uctbx
from cctbx.miller import index_generator

from interceptor import packagefinder, read_config_file
from iota.base.processor import Processor, phil_scope as dials_scope
from iota.utils.utils import Capturing

from interceptor.format import extract_data

#AI Stuff
from resonet.utils.predict_fabio import ImagePredictFabio


# Custom PHIL for processing with DIALS stills processor
custom_param_string = """
output {
  experiments_filename = None
  indexed_filename = None
  strong_filename = None
  refined_experiments_filename = None
  integrated_experiments_filename = None
  integrated_filename = None
  profile_filename = None
  integration_pickle = None
  composite_output = False
}
spotfinder {
  filter {
    max_spot_size = 1000
  }
  threshold {
    algorithm = *dispersion dispersion_extended
    dispersion {
      gain = 1
      global_threshold = 0
    }
  }
}
indexing {
  refinement_protocol {
    d_min_start = 2.0
  }
  stills {
    indexer = Auto *stills sequences
    method_list = fft3d fft1d real_space_grid_search
  }
  basis_vector_combinations {
    max_combinations = 10
  }
}
integration {
  background {
    simple {
      outlier {
        algorithm = *null nsigma truncated normal plane tukey
      }
    }
  }
}
significance_filter {
  enable = True
  d_min = None
  n_bins = 20
  isigi_cutoff = 1.0
}
"""
custom_phil = ip.parse(custom_param_string)
custom_params = dials_scope.fetch(source=custom_phil).extract()


class MemReaderNamedPath(MemReader):

  def __init__(self, path,  *args, **kwargs):
    self.dummie_path_name = path
    super(MemReaderNamedPath, self).__init__(*args, **kwargs)


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

        # Check for custom models and architectures
        reso_model = self.cfg.getstr('resolution_model')
        reso_arch = self.cfg.getstr('resolution_architecture')
        multi_model = self.cfg.getstr('multilattice_model')
        multi_arch = self.cfg.getstr('multilattice_architecture')

        # Generate predictor
        assert reso_model is not None
        assert multi_model is not None
        self.predictor = ImagePredictFabio(
            reso_model=reso_model,
            reso_arch=reso_arch,
            multi_model=multi_model,
            multi_arch=multi_arch,
            ice_model=None,
            ice_arch=None
        )

    @staticmethod
    def d_to_dnew(d):
        B = 4 * d ** 2 + 12
        # new equation: B = 13*dnew^2 -23 *dnew + 29
        # quadratic fit coef
        a, b, c = 13., -22., 26. - B
        dnew = .5 * (-b + np.sqrt(b ** 2 - 4 * a * c)) / a  # positive root
        return dnew

    def estimate_resolution(self):
        res = self.predictor.detect_resolution()
        if self.cfg.getboolean('use_modern_res_trend'):
            res = self.d_to_dnew(res)
        return res

    def find_rings(self):
        return 0

    def find_splitting(self):
        return self.predictor.detect_multilattice_scattering()

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
        self.split = self.find_splitting()
        return score


class ImageScorer(object):
    def __init__(self, config, experiments=None, observed=None):
        self.cfg = config
        if experiments and observed:
            self.initialize(experiments, observed)

    def initialize(self, experiments, observed):
        self.experiments = experiments
        self.observed = observed

        # extract reflections and map to reciprocal space
        self.refl = observed.select(observed["id"] == 0)
        self.refl.centroid_px_to_mm([experiments[0]])
        self.refl.map_centroids_to_reciprocal_space([experiments[0]])

        # calculate d-spacings
        self.ref_d_star_sq = flex.pow2(self.refl["rlp"].norms())
        self.d_spacings = uctbx.d_star_sq_as_d(self.ref_d_star_sq)
        self.d_min = flex.min(self.d_spacings)

        # initialize desired parameters (so that can back out without having to
        # re-run functions)
        self.n_ice_rings = 0
        self.mean_spot_shape_ratio = 1
        self.n_overloads = self.count_overloads()
        self.hres = 99.9
        self.n_spots = 0

    def filter_by_resolution(self, refl, d_min, d_max):
        d_min = float(d_min) if d_min is not None else 0.1
        d_max = float(d_max) if d_max is not None else 99

        d_star_sq = flex.pow2(refl["rlp"].norms())
        d_spacings = uctbx.d_star_sq_as_d(d_star_sq)
        filter = flex.bool(len(d_spacings), False)
        filter = filter | (d_spacings >= d_min) & (d_spacings <= d_max)
        return filter

    def calculate_stats(self, verbose=False):
        # Only accept "good" spots based on specific parameters, if selected

        # 1. No ice
        if self.cfg.getboolean("spf_ice_filter"):
            ice_sel = per_image_analysis.ice_rings_selection(self.refl)
            spots_no_ice = self.refl.select(~ice_sel)
        else:
            spots_no_ice = self.refl

        # 2. Falls between 40 - 4.5A
        if self.cfg.getboolean('spf_good_spots_only'):
            res_lim_sel = self.filter_by_resolution(
                refl=spots_no_ice,
                d_min=self.cfg.getstr('spf_d_min'),
                d_max=self.cfg.getstr('spf_d_max'))
            good_spots = spots_no_ice.select(res_lim_sel)
        else:
            good_spots = spots_no_ice

        # Saturation / distance are already filtered by the spotfinder
        self.n_spots = good_spots.size()

        # Estimate resolution by spots that aren't ice (susceptible to poor ice ring
        # location)
        if self.n_spots > 10:
            self.hres = per_image_analysis.estimate_resolution_limit(spots_no_ice)
        else:
            self.hres = 99.9

        if verbose:
            print('SCORER: no. spots total = ', self.refl.size())
            if self.cfg.getboolean('spf_ice_filter'):
                print('SCORER: no. spots (no ice) = ', spots_no_ice.size())
                no_ice = 'no ice, '
            else:
                no_ice = ''
            if self.cfg.getboolean('spf_good_spots_only'):
                print('SCORER: no. spots ({}w/in res limits) = '.format(no_ice),
                      good_spots.size())

    def count_overloads(self):
        """ A function to determine the number of overloaded spots """
        overloads = [i for i in self.observed.is_overloaded(self.experiments) if i]
        return len(overloads)

    def count_ice_rings(self, width=0.002, verbose=False):
        """ A function to find and count ice rings (modeled after
        dials.algorithms.integration.filtering.PowderRingFilter, with some alterations:
            1. Hard-coded with ice unit cell / space group
            2. Returns spot counts vs. water diffraction resolution "bin"

        Rather than finding ice rings themselves (which may be laborious and time
        consuming), this method relies on counting the number of found spots that land
        in regions of water diffraction. A weakness of this approach is that if any spot
        filtering or spot-finding parameters are applied by prior methods, not all ice
        rings may be found. This is acceptable, since the purpose of this method is to
        determine if water and protein diffraction occupy the same resolutions.
        """
        ice_start = time.time()

        unit_cell = uctbx.unit_cell((4.498, 4.498, 7.338, 90, 90, 120))
        space_group = sgtbx.space_group_info(number=194).group()

        # Correct unit cell
        unit_cell = space_group.average_unit_cell(unit_cell)

        half_width = width / 2
        d_min = uctbx.d_star_sq_as_d(uctbx.d_as_d_star_sq(self.d_min) + half_width)

        # Generate a load of indices
        generator = index_generator(unit_cell, space_group.type(), False, d_min)
        indices = generator.to_array()

        # Compute d spacings and sort by resolution
        d_star_sq = flex.sorted(unit_cell.d_star_sq(indices))
        d = uctbx.d_star_sq_as_d(d_star_sq)
        dd = list(zip(d_star_sq, d))

        # identify if spots fall within ice ring areas
        results = []
        for ds2, d_res in dd:
            result = [i for i in (flex.abs(self.ref_d_star_sq - ds2) < half_width) if i]
            results.append((d_res, len(result)))

        possible_ice = [r for r in results if r[1] / len(self.observed) * 100 >= 5]

        if verbose:
            print(
                "SCORER: ice ring time = {:.5f} seconds".format(time.time() - ice_start)
            )

        self.n_ice_rings = len(possible_ice)  # output in info

        return self.n_ice_rings

    def spot_elongation(self, verbose=False):
        """ Calculate how elongated spots are on average (i.e. shoebox axis ratio).
            Only using x- and y-axes for this calculation, assuming stills and z = 1
        :return: elong_mean = mean elongation ratio
                 elong_median = median elongation ratio
                 elong_std = standard deviation of elongation ratio
        """
        e_start = time.time()
        axes = [self.observed[i]["shoebox"].size() for i in range(len(self.observed))]

        elong = [np.max((x, y)) / np.min((x, y)) for z, y, x in axes]
        elong_mean = np.mean(elong)
        elong_median = np.median(elong)
        elong_std = np.std(elong)

        if verbose:
            print(
                "SCORER: spot shape time = {:.5f} seconds".format(time.time() - e_start)
            )

        # for output reporting
        self.mean_spot_shape_ratio = elong_mean

        return elong_mean, elong_median, elong_std

    def find_max_intensity(self, verbose=False):
        """ Determine maximum intensity among reflections between 15 and 4 A
        :return: max_intensity = maximum intensity between 15 and 4 A
        """
        max_start = time.time()

        sel_inbounds = flex.bool(len(self.d_spacings), False)
        sel_inbounds = sel_inbounds | (self.d_spacings >= 4) & (self.d_spacings <= 15)
        refl_inbounds = self.refl.select(sel_inbounds)
        intensities = [
            refl_inbounds[i]["intensity.sum.value"] for i in range(len(refl_inbounds))
        ]

        if verbose:
            print(
                "SCORER: max intensity time = {:.5f} seconds".format(
                    time.time() - max_start
                )
            )

        if intensities:
            return np.max(intensities)
        else:
            return 0

    def calculate_score(self, verbose=False):
        """ This *more or less* replicates the scoring approach from libdistl
        :param experiments: ExperimentList object
        :param observed: Found spots
        :param hres: high resolution limit of found spots
        :param score: starting score
        :param verbose: Outputs scoring details to stdout
        :return: score = final score
        """
        score = 0

        # Calculate # of spots and resolution here
        self.calculate_stats(verbose=verbose)

        # calculate score by resolution using heuristic
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

        if verbose:
            print("SCORER: resolution = {:.2f}, score = {}".format(self.hres, score))

        # calculate "diffraction strength" from maximum pixel value
        max_I = self.find_max_intensity(verbose=verbose)
        if max_I >= 40000:
            score += 2
        elif 40000 > max_I >= 15000:
            score += 1

        if verbose:
            print("SCORER: max intensity (15-4Å) = {}, score = {}".format(max_I, score))

        # evaluate ice ring presence
        n_ice_rings = self.count_ice_rings(width=0.02, verbose=verbose)
        if n_ice_rings >= 4:
            score -= 3
        elif 4 > n_ice_rings >= 2:
            score -= 2
        elif n_ice_rings == 1:
            score -= 1

        if verbose:
            print("SCORER: {} ice rings found, score = {}".format(n_ice_rings, score))

        # bad spot penalty, good spot boost
        e_mean, e_median, e_std = self.spot_elongation(verbose=verbose)
        if e_mean > 2.0:
            score -= 2
        if e_std > 1.0:
            score -= 2
        if e_median < 1.35 and e_std < 0.4:
            score += 2

        if verbose:
            print(
                "SCORER: spot shape ratio mean = {:.2f}, "
                "median = {:.2f}, std = {:.2f}; score = {}".format(
                    e_mean, e_median, e_std, score
                )
            )

        if score <= 0:
            if self.hres > 20:
                score = 0
            else:
                score = 1

        if verbose:
            print("SCORER: final score = {}".format(score))
            print("SCORER: {} overloaded reflections".format(self.n_overloads))

        return score


class InterceptorBaseProcessor(object):
    def __init__(self, run_mode='DEFAULT', configfile=None, test=False):
        self.processing_mode = 'spotfinding'
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

        # Generate DIALS Stills Processor params
        params, self.dials_phil = self.generate_params()

        # Instantiate Stills Processor
        self.processor = Processor(params=params)
        self.processor.write_pickle = False # todo: set this in IOTA base processor

    def generate_params(self):
        # read in DIALS settings from PHIL file
        if self.cfg.getstr('processing_phil_file'):
            with open(self.cfg.getstr('processing_phil_file'), "r") as pf:
                target_phil = ip.parse(pf.read())
        else:
            target_phil = ip.parse(custom_param_string)
        new_phil = dials_scope.fetch(source=target_phil)
        params = new_phil.extract()
        return params, new_phil

    def print_params(self):
        print("\nParameters for this run: ")
        diff_phil = dials_scope.fetch_diff(source=self.dials_phil)
        diff_phil.show()
        print("\n")

    @staticmethod
    def make_experiments(filename=None, data=None, detector=None):
        # make experiments
        e_start = time.time()
        if data is None:
            # Regular ExperimentList creation with regular files
            assert filename is not None
            experiments = ExperimentListFactory.from_filenames([filename])
            e_time = time.time() - e_start
            return experiments, e_time

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
            info['n_spots'] = 0
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


class FileProcessor(InterceptorBaseProcessor):
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

    def process(self, filename, info):
        info["phil"] = self.dials_phil.as_str()

        # Make ExperimentList object
        experiments, e_time = self.make_experiments(filename=filename)

        # Spotfinding
        start = time.time()
        try:
            observed = self.processor.find_spots(experiments)
        except Exception as err:
            import traceback
            spf_tb = traceback.format_exc()
            info["spf_error"] = "SPF ERROR: {}".format(str(err))
            info['spf_error_tback'] = spf_tb
            return info
        else:
            if observed.size() >= self.cfg.getint('min_Bragg_peaks'):
                scorer = ImageScorer(experiments, observed, config=self.cfg)
                try:
                    if self.cfg.getboolean('spf_calculate_score'):
                        info["score"] = scorer.calculate_score(verbose=self.verbose)
                    else:
                        info["score"] = -999
                        scorer.calculate_stats()
                except Exception as e:
                    info["n_spots"] = 0
                    info["scr_error"] = "SCORING ERROR: {}".format(e)
                else:
                    info["n_spots"] = scorer.n_spots
                    info["hres"] = scorer.hres
                    info["n_ice_rings"] = scorer.n_ice_rings
                    info["n_overloads"] = scorer.n_overloads
                    info["mean_shape_ratio"] = scorer.mean_spot_shape_ratio
            else:
                info["n_spots"] = observed.size()
        print("SCORING TIME = {}".format(time.time() - start))


        # Doing it here because scoring can reject spots within ice rings, which can
        # drop the number below the minimal limit
        if info['n_spots'] < self.cfg.getint('min_Bragg_peaks'):
            info[
                "spf_error"] = "Too few ({}) spots found!".format(
                observed.size())

        # if last stage was selected to be "spotfinding", stop here; otherwise
        # perform a speed check before proceeding to indexing
        if self.cfg.getstr('processing_mode') == "spotfinding" or info["n_spots"] <= \
                self.cfg.getint('min_Bragg_peaks'):
            return info
        else:
            exposure_time_cutoff = self.cfg.getfloat('exposure_time_cutoff')
            if exposure_time_cutoff > info['exposure_time']:
                return info

        # Indexing
        with Capturing() as idx_output:
            try:
                experiments, indexed = self.processor.index(experiments, observed)
                solution = self.processor.refine_bravais_settings(indexed, experiments)
                if solution is not None:
                    experiments, indexed = self.processor.reindex(indexed, experiments,
                                                        solution)
                else:
                    info[
                        "rix_error"] = "reindex error: symmetry solution not found!"
                if len(indexed) == 0:
                    info["idx_error"] = "index error: no indexed reflections!"
            except Exception as err:
                info["idx_error"] = "index error: {}".format(str(err))
                return info
            else:
                if indexed:
                    lat = experiments[0].crystal.get_space_group().info()
                    sg = str((lat)).replace(" ", "")
                    unit_cell = experiments[0].crystal.get_unit_cell().parameters()
                    uc = " ".join(["{:.2f}".format(i) for i in unit_cell])
                    info["n_indexed"] = len(indexed)
                    info["sg"] = sg
                    info["uc"] = uc

        # if last step was 'indexing', stop here
        if "index" in self.cfg.getstr('processing_mode'):
            return info

        # **** Integration **** #
        with Capturing() as int_output:
            try:
                experiments, indexed = self.processor.refine(experiments, indexed)
                integrated = self.processor.integrate(experiments, indexed)
            except Exception as err:
                info["int_error"] = "integration error: {}".format(str(err))
                return info
            else:
                info["int_error"] = '{} integrated'.format(integrated.size())
                if integrated:
                    info['n_integrated'] = integrated.size()
                return info

    def run(self, filename, info):
        start = time.time()
        info = self.process(filename, info)
        info['proc_time'] = time.time() - start
        return info


class ZMQProcessor(InterceptorBaseProcessor):
    def __init__(
            self,
            run_mode='DEFAULT',
            configfile=None,
            detectorfile=None,
            test=False,
    ):
        InterceptorBaseProcessor.__init__(self, run_mode=run_mode,
                                          configfile=configfile,
                                          test=test)
        self.ai_scorer = AIScorer(config=self.cfg)

        try:
            self.sp_scorer = ImageScorer(config=self.cfg)
        except AssertionError:
            self.sp_scorer = None

    def process(self, data, detector, info):
        #info["phil"] = self.dials_phil.as_str()
        filename = info['full_path']

        # Make ExperimentList object
        experiments, e_time = self.make_experiments(data=data, detector=detector, filename=filename)

        # Spotfinding
        with Capturing() as spf_output:
            try:
                observed = self.processor.find_spots(experiments)
            except Exception as err:
                # import traceback
                # spf_tb = traceback.format_exc()
                info["spf_error"] = "SPF ERROR: {}".format(err)
                # info['spf_error_tback'] = spf_tb
                return info
            else:
                if observed.size() >= self.cfg.getint('min_Bragg_peaks'):
                    self.sp_scorer.initialize(experiments=experiments, observed=observed)
                    try:
                        if self.cfg.getboolean('spf_calculate_score'):
                            info["score"] = self.sp_scorer.calculate_score()
                        else:
                            info["score"] = -999
                            self.sp_scorer.calculate_stats()
                    except Exception as e:
                        info["n_spots"] = 0
                        info["scr_error"] = "SCORING ERROR: {}".format(e)
                    else:
                        info["n_spots"] = self.sp_scorer.n_spots
                        info["hres"] = self.sp_scorer.hres
                        info["n_ice_rings"] = self.sp_scorer.n_ice_rings
                        info["n_overloads"] = self.sp_scorer.n_overloads
                        info["mean_shape_ratio"] = self.sp_scorer.mean_spot_shape_ratio
                else:
                    info["n_spots"] = observed.size()

            # Perform additional analysis via AI (note: this will take over the whole process someday)
            if self.cfg.getboolean('use_ai'):
                if self.sp_scorer is None:
                    info['ai_error'] = 'AI_ERROR: XRAIS FAILED TO INITIALIZE'
                    return info
                try:
                    encoding_info = json.loads(data.get("streamfile_2", ""))
                    raw_bytes = data.get("streamfile_3", "")
                    header = json.loads(data.get("header2", ""))

                    raw_data = extract_data(info=encoding_info, data=raw_bytes)

                    self.ai_scorer.predictor.load_image_from_file_or_array(
                        raw_image=raw_data,
                        detdist=header['detector_distance'] * 1000,
                        pixsize=header['x_pixel_size'] * 1000,
                        wavelen=header['wavelength'],
                    )
                    score = self.ai_scorer.calculate_score()  # Once the score works, will replace
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

        # Doing it here because scoring can reject spots within ice rings, which can
        # drop the number below the minimal limit
        if info['n_spots'] < self.cfg.getint('min_Bragg_peaks'):
            info["spf_error"] = "Too few ({}) spots found!".format(
                observed.size())

        # if last stage was selected to be "spotfinding", stop here; otherwise
        # perform a speed check before proceeding to indexing
        if self.cfg.getstr('processing_mode') == "spotfinding" or info["n_spots"] <= \
                self.cfg.getint('min_Bragg_peaks'):
            return info
        else:
            exposure_time_cutoff = self.cfg.getfloat('exposure_time_cutoff')
            if exposure_time_cutoff > info['exposure_time']:
                return info

        # Indexing
        with Capturing() as idx_output:
            try:
                experiments, indexed = self.processor.index(experiments, observed)
                solution = self.processor.refine_bravais_settings(indexed, experiments)
                if solution is not None:
                    experiments, indexed = self.processor.reindex(indexed, experiments, solution)
                else:
                    info["rix_error"] = "reindex error: symmetry solution not found!"
                if len(indexed) == 0:
                    info["idx_error"] = "index error: no indexed reflections!"
            except Exception as err:
                info["idx_error"] = "index error: {}".format(str(err))
                return info
            else:
                if indexed:
                    lat = experiments[0].crystal.get_space_group().info()
                    sg = str((lat)).replace(" ", "")
                    unit_cell = experiments[0].crystal.get_unit_cell().parameters()
                    uc = " ".join(["{:.2f}".format(i) for i in unit_cell])
                    info["n_indexed"] = len(indexed)
                    info["sg"] = sg
                    info["uc"] = uc

        # if last step was 'indexing', stop here
        if "index" in self.cfg.getstr('processing_mode'):
            return info

    def run(self, data, detector, info):
        return self.process(data, detector, info)


def calculate_score(experiments, observed):
    start = time.time()
    scorer = ImageScorer(experiments, observed)
    print("scoring init: {:.5f} seconds".format(time.time() - start))
    score_start = time.time()
    scorer.calculate_score(verbose=True)
    print("scoring time: {:.5f} seconds".format(time.time() - score_start))
    print("total time: {:.5f} seconds".format(time.time() - start))

if __name__ == "__main__":
    proc = ZMQProcessor()
    proc.dials_phil.show()

# -- end
