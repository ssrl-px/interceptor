from __future__ import absolute_import, division, print_function

'''
Author      : Lyubimov, A.Y.
Created     : 03/31/2020
Last Changed: 03/31/2020
Description : Streaming stills processor for live data analysis
'''

import copy
from cctbx import sgtbx
from iotbx import phil as ip

from dials.command_line.stills_process import Processor, \
  phil_scope as dials_scope
from dials.command_line.refine_bravais_settings import phil_scope as sg_scope, \
  bravais_lattice_to_space_group_table
from dials.algorithms.indexing.bravais_settings import \
  refined_settings_from_refined_triclinic
from dials.algorithms.spot_finding import per_image_analysis
from dials.array_family import flex

from iota.components.iota_utils import Capturing

# Custom PHIL for processing with DIALS stills processor
custom_params = '''
output {
  experiments_filename = None
  indexed_filename = None
  strong_filename = None
  refined_experiments_filename = None
  integrated_experiments_filename = None
  integrated_filename = None
  profile_filename = None  
}
spotfinder {
  threshold {
    dispersion {
      gain = 1
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
'''


class FastProcessor(Processor):
  def __init__(self, last_stage='spotfinding', min_Bragg=10):
    self.last_stage = last_stage
    self.min_Bragg = min_Bragg
    params, self.dials_phil = self.generate_params()
    Processor.__init__(self, params=params)

    # diff_phil = dials_scope.fetch_diff(source=self.dials_phil)
    # diff_phil.show()

    self.params.indexing.stills.method_list = [
      # 'fft3d',
      'fft1d',
      'real_space_grid_search']
    self.params.significance_filter.enable = True
    self.params.significance_filter.isigi_cutoff = 1

  def generate_params(self):
    phil = dials_scope.fetch(source=ip.parse(custom_params))
    params = phil.extract()
    return params, phil

  def refine_bravais_settings(self, reflections, experiments):
    sgparams = sg_scope.fetch(self.dials_phil).extract()
    sgparams.refinement.reflections.outlier.algorithm = 'tukey'
    crystal_P1 = copy.deepcopy(experiments[0].crystal)

    try:
      refined_settings = refined_settings_from_refined_triclinic(
        experiments=experiments,
        reflections=reflections,
        params=sgparams
      )
      possible_bravais_settings = {s["bravais"] for s in refined_settings}
      bravais_lattice_to_space_group_table(possible_bravais_settings)
    except Exception:
      for expt in experiments:
        expt.crystal = crystal_P1
      return None

    lattice_to_sg_number = {
      'aP': 1, 'mP': 3, 'mC': 5, 'oP': 16, 'oC': 20, 'oF': 22, 'oI': 23,
      'tP': 75, 'tI': 79, 'hP': 143, 'hR': 146, 'cP': 195, 'cF': 196, 'cI': 197
    }
    filtered_lattices = {}
    for key, value in lattice_to_sg_number.items():
      if key in possible_bravais_settings:
        filtered_lattices[key] = value

    highest_sym_lattice = max(filtered_lattices, key=filtered_lattices.get)
    highest_sym_solutions = [s for s in refined_settings
                             if s['bravais'] == highest_sym_lattice]
    if len(highest_sym_solutions) > 1:
      highest_sym_solution = sorted(highest_sym_solutions,
                                    key=lambda x: x['max_angular_difference'])[0]
    else:
      highest_sym_solution = highest_sym_solutions[0]

    return highest_sym_solution

  def reindex(self, reflections, experiments, solution):
    """ Reindex with newly-determined space group / unit cell """

    # Update space group / unit cell
    experiment = experiments[0]
    experiment.crystal.update(solution.refined_crystal)

    # Change basis
    cb_op = solution['cb_op_inp_best'].as_abc()
    change_of_basis_op = sgtbx.change_of_basis_op(cb_op)
    miller_indices = reflections['miller_index']
    non_integral_indices = change_of_basis_op.apply_results_in_non_integral_indices(miller_indices)
    sel = flex.bool(miller_indices.size(), True)
    sel.set_selected(non_integral_indices, False)
    miller_indices_reindexed = change_of_basis_op.apply(miller_indices.select(sel))
    reflections['miller_index'].set_selected(sel, miller_indices_reindexed)
    reflections['miller_index'].set_selected(~sel, (0, 0, 0))

    return experiments, reflections

  def pg_and_reindex(self, indexed, experiments):
    ''' Find highest-symmetry Bravais lattice '''
    solution = self.refine_bravais_settings(indexed, experiments)
    if solution is not None:
      experiments, indexed = self.reindex(indexed, experiments, solution)
      return experiments, indexed, 'success'
    else:
      return experiments, indexed, 'failed'

  def calculate_resolution_from_spotfinding(self, observed, experiments):
    # get detector and beam
    try:
      detector = experiments.unique_detectors()[0]
      beam = experiments.unique_beams()[0]
    except AttributeError:
      detector = experiments.imagesets()[0].get_detector()
      beam = experiments.imagesets()[0].get_beam()

    s1 = flex.vec3_double()
    for i in range(len(observed)):
      obs_px_value   =  observed['xyzobs.px.value'][i][0:2]
      obs_location   =  detector[observed['panel'][i]]
      obs_coord      =  obs_location.get_pixel_lab_coord(obs_px_value)
      s1.append(obs_coord)
    two_theta = s1.angle(beam.get_s0())
    d = beam.get_wavelength() / (2 * flex.asin(two_theta / 2))
    return np.max(d), np.min(d)

  def process(self, experiments, info):

    # Spotfinding
    with Capturing() as spf_output:
      try:
        observed = self.find_spots(experiments)
        if len(observed) == 0:
          info['spf_error'] = 'spotfinding error: no spots found!'
      except Exception as err:
        info['spf_error'] = 'spotfinding error: {}'.format(str(err))
        return info
      else:
        experiment = experiments[0]
        refl = observed.select(observed["id"] == 0)
        refl.centroid_px_to_mm([experiment])
        refl.map_centroids_to_reciprocal_space([experiment])
        stats = per_image_analysis.stats_per_image(experiment, refl)
        info['n_spots'] = stats.n_spots_no_ice[0]
        info['hres'] = stats.estimated_d_min[0]

        # Get (and print) information from experiments
        try:
          imgset = experiments.imagesets()[0]
          beam = imgset.get_beam()
          s0 = beam.get_s0()
          detector = imgset.get_detector()[0]
          info['beamXY'] = detector.get_beam_centre_px(s0)
          info['dist'] = detector.get_distance()
        except Exception as err:
          info['img_error'] = 'Could not get beam center coords: {}' \
                              ''.format(str(err))

    # if last stage was selected to be "spotfinding", stop here
    if self.last_stage == 'spotfinding' or info['n_spots'] <= self.min_Bragg:
      return info

    # Indexing
    with Capturing() as idx_output:
      try:
        experiments, indexed = self.index(experiments, observed)
        solution = self.refine_bravais_settings(indexed, experiments)
        if solution is not None:
          experiments, indexed = self.reindex(indexed, experiments, solution)
        else:
          info["rix_error"] = 'reindex error: symmetry solution not found!'
        if len(indexed) == 0:
          info['idx_error'] = 'index error: no indexed reflections!'
      except Exception as err:
        info['idx_error'] = 'index error: {}'.format(str(err))
        return info
      else:
        if indexed:
          lat = experiments[0].crystal.get_space_group().info()
          sg = str((lat)).replace(' ', '')
          unit_cell = experiments[0].crystal.get_unit_cell().parameters()
          uc = ' '.join(['{:.2f}'.format(i) for i in unit_cell])
          info['n_indexed'] = len(indexed)
          info['sg'] = sg
          info['uc'] = uc

    # if last step was 'indexing', stop here
    if 'index' in self.last_stage:
      return info

  def run(self, experiments, info):
    return self.process(experiments, info)
