from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 05/15/2020
Description : Streaming stills processor for live data analysis
"""
import time  # noqa: F401; keep around for testing

import numpy as np

from cctbx import sgtbx, crystal
from iotbx import phil as ip
from spotfinder.array_family import flex

from dials.algorithms.spot_finding import per_image_analysis
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory

from cctbx import uctbx
from cctbx.miller import index_generator

from interceptor import packagefinder, read_config_file
from iota.base.processor import Processor, phil_scope as dials_scope
from iota.utils.utils import Capturing

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


class ImageScorer(object):
    def __init__(self, experiments, observed, config):
        self.experiments = experiments
        self.observed = observed
        self.cfg = config

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
            print(
                "SCORER: max intensity (15-4Å) = {}, score = {}".format(max_I,
                                                                           score))

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
    def make_experiments(filename, data=None, detector=None):
        # make experiments
        e_start = time.time()
        if data:
            if 'eiger' in detector.lower():
                from interceptor.format import FormatEigerStream
                FormatEigerStream.injected_data = data
            elif 'pilatus' in detector.lower():
                from interceptor.format import FormatPilatusStream
                FormatPilatusStream.injected_data = data
            experiments = ExperimentListFactory.from_filenames([filename])
        else:
            experiments = ExperimentListFactory.from_filenames([filename])
        e_time = time.time() - e_start
        return experiments, e_time


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
        experiments, e_time = self.make_experiments(filename)

        # Spotfinding
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
            detector='EIGER',
            configfile=None,
            test=False,
    ):
        self.detector = detector
        InterceptorBaseProcessor.__init__(self, run_mode=run_mode,
                                          configfile=configfile, test=test)

    def process(self, data, filename, info):
        info["phil"] = self.dials_phil.as_str()

        # Make ExperimentList object
        experiments, e_time = self.make_experiments(data=data, filename=filename, detector=self.detector)

        # Spotfinding
        with Capturing() as spf_output:
            try:
                observed = self.processor.find_spots(experiments)
            except Exception as err:
                import traceback
                spf_tb = traceback.format_exc()
                #DEBUG - CHANGE TO ERR
                info["spf_error"] = "SPF ERROR: {}".format(len(experiments))
                info['spf_error_tback'] = spf_tb
                return info
            else:
                if observed.size() >= self.cfg.getint('min_Bragg_peaks'):
                    scorer = ImageScorer(experiments, observed, config=self.cfg)
                    try:
                        if self.cfg.getboolean('spf_calculate_score'):
                            info["score"] = scorer.calculate_score()
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

    def run(self, data, filename, info):
        return self.process(data, filename, info)


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
