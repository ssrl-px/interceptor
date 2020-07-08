from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 05/15/2020
Description : Streaming stills processor for live data analysis
"""

import copy
import time  # noqa: F401; keep around for testing

import numpy as np

from cctbx import sgtbx
from iotbx import phil as ip

from dials.command_line.stills_process import Processor, phil_scope as dials_scope
from dials.command_line.refine_bravais_settings import (
    phil_scope as sg_scope,
    bravais_lattice_to_space_group_table,
)
from dials.algorithms.indexing.bravais_settings import (
    refined_settings_from_refined_triclinic,
)
from dials.algorithms.spot_finding import per_image_analysis
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory

from cctbx import uctbx
from cctbx.miller import index_generator

from interceptor import packagefinder, read_config_file, Processing_config
from interceptor.format import FormatEigerStreamSSRL
from iota.components.iota_utils import Capturing

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


def make_experiments(data, filename):
    # make experiments
    e_start = time.time()
    FormatEigerStreamSSRL.inject_data(data)
    experiments = ExperimentListFactory.from_filenames([filename])
    e_time = time.time() - e_start
    return experiments, e_time


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
        if d_min is None:
            d_min = 0.1
        if d_max is None:
            d_max = 99
        d_star_sq = flex.pow2(refl["rlp"].norms())
        d_spacings = uctbx.d_star_sq_as_d(d_star_sq)
        filter = flex.bool(len(d_spacings), False)
        filter = filter | (d_spacings >= d_min) & (d_spacings <= d_max)
        return filter

    def calculate_stats(self, verbose=False):
        # Only accept "good" spots based on specific parameters, if selected

        # 1. No ice
        if self.cfg.spf_ice_filter:
            ice_sel = per_image_analysis.ice_rings_selection(self.refl)
            spots_no_ice = self.refl.select(~ice_sel)
        else:
            spots_no_ice = self.refl

        # 2. Falls between 40 - 4.5A
        if self.cfg.spf_good_spots_only:
            res_lim_sel = self.filter_by_resolution(
                refl=spots_no_ice,
                d_min=self.cfg.spf_d_min,
                d_max=self.cfg.spf_d_max)
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
            if self.cfg.spf_ice_filter:
                print('SCORER: no. spots (no ice) = ', spots_no_ice.size())
                no_ice = 'no ice, '
            else:
                no_ice = ''
            if self.cfg.spf_good_spots_only:
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

        return np.max(intensities)

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
        n_ice_rings = self.count_ice_rings(verbose=verbose)
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


class FastProcessor(Processor):
    def __init__(
            self,
            run_mode='DEFAULT',
            configfile=None,
            test=False,
    ):
        self.processing_mode = 'spotfinding'
        self.test = test
        self.run_mode = None

        # generate processing config params
        if configfile:
            p_config = read_config_file(configfile)
        else:
            p_config = packagefinder('processing.cfg', 'connector', read_config=True)
        config_dict = {k:v for k, v in p_config[run_mode].items()}
        self.cfg = Processing_config(**config_dict)

        # Generate DIALS Stills Processor params
        params, self.dials_phil = self.generate_params()

        # Initialize Stills Processor
        Processor.__init__(self, params=params)

    def generate_params(self):
        # read in DIALS settings from PHIL file
        if self.cfg.processing_phil_file:
            with open(self.cfg.processing_phil_file, "r") as pf:
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

    def refine_bravais_settings(self, reflections, experiments):
        sgparams = sg_scope.fetch(self.dials_phil).extract()
        sgparams.refinement.reflections.outlier.algorithm = "tukey"
        crystal_P1 = copy.deepcopy(experiments[0].crystal)

        try:
            refined_settings = refined_settings_from_refined_triclinic(
                experiments=experiments, reflections=reflections, params=sgparams
            )
            possible_bravais_settings = {s["bravais"] for s in refined_settings}
            bravais_lattice_to_space_group_table(possible_bravais_settings)
        except Exception:
            for expt in experiments:
                expt.crystal = crystal_P1
            return None

        lattice_to_sg_number = {
            "aP": 1,
            "mP": 3,
            "mC": 5,
            "oP": 16,
            "oC": 20,
            "oF": 22,
            "oI": 23,
            "tP": 75,
            "tI": 79,
            "hP": 143,
            "hR": 146,
            "cP": 195,
            "cF": 196,
            "cI": 197,
        }
        filtered_lattices = {}
        for key, value in lattice_to_sg_number.items():
            if key in possible_bravais_settings:
                filtered_lattices[key] = value

        highest_sym_lattice = max(filtered_lattices, key=filtered_lattices.get)
        highest_sym_solutions = [
            s for s in refined_settings if s["bravais"] == highest_sym_lattice
        ]
        if len(highest_sym_solutions) > 1:
            highest_sym_solution = sorted(
                highest_sym_solutions, key=lambda x: x["max_angular_difference"]
            )[0]
        else:
            highest_sym_solution = highest_sym_solutions[0]

        return highest_sym_solution

    def reindex(self, reflections, experiments, solution):
        """ Reindex with newly-determined space group / unit cell """

        # Update space group / unit cell
        experiment = experiments[0]
        experiment.crystal.update(solution.refined_crystal)

        # Change basis
        cb_op = solution["cb_op_inp_best"].as_abc()
        change_of_basis_op = sgtbx.change_of_basis_op(cb_op)
        miller_indices = reflections["miller_index"]
        non_integral_indices = change_of_basis_op.apply_results_in_non_integral_indices(
            miller_indices
        )
        sel = flex.bool(miller_indices.size(), True)
        sel.set_selected(non_integral_indices, False)
        miller_indices_reindexed = change_of_basis_op.apply(miller_indices.select(sel))
        reflections["miller_index"].set_selected(sel, miller_indices_reindexed)
        reflections["miller_index"].set_selected(~sel, (0, 0, 0))

        return experiments, reflections

    def pg_and_reindex(self, indexed, experiments):
        """ Find highest-symmetry Bravais lattice """
        solution = self.refine_bravais_settings(indexed, experiments)
        if solution is not None:
            experiments, indexed = self.reindex(indexed, experiments, solution)
            return experiments, indexed, "success"
        else:
            return experiments, indexed, "failed"

    def process(self, data, filename, info):
        info["phil"] = self.dials_phil.as_str()

        # Make ExperimentList object
        experiments, e_time = make_experiments(data, filename)

        # Spotfinding
        with Capturing() as spf_output:
            try:
                observed = self.find_spots(experiments)
            except Exception as err:
                info["spf_error"] = "spotfinding error: {}".format(str(err))
                return info
            else:
                if observed.size() > 10 and self.cfg.spf_calculate_score:
                    try:
                        scorer = ImageScorer(experiments, observed, config=self.cfg)
                        info["score"] = scorer.calculate_score()
                        info["n_spots"] = scorer.n_spots
                        info["hres"] = scorer.hres
                        info["n_ice_rings"] = scorer.n_ice_rings
                        info["n_overloads"] = scorer.n_overloads
                        info["mean_shape_ratio"] = scorer.mean_spot_shape_ratio
                    except Exception as e:
                        info["n_spots"] = observed.size()
                        info["scr_error"] = "scoring error: {}".format(e)
                else:
                    info["n_spots"] = observed.size()
                    info[
                        "spf_error"] = "spotfinding error: insufficient spots found ({})!".format(
                        observed.size())

        # if last stage was selected to be "spotfinding", stop here
        if self.cfg.processing_mode == "spotfinding" or info["n_spots"] <= \
                self.min_Bragg:
            return info

        # Indexing
        with Capturing() as idx_output:
            try:
                experiments, indexed = self.index(experiments, observed)
                solution = self.refine_bravais_settings(indexed, experiments)
                if solution is not None:
                    experiments, indexed = self.reindex(indexed, experiments, solution)
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
        if "index" in self.cfg.processing_mode:
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
    proc = FastProcessor()
    proc.dials_phil.show()

# -- end
