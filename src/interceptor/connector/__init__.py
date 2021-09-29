import os
import sys
from dxtbx.model.experiment_list import ExperimentListFactory as ExLF
from dials.array_family import flex
from iotbx import phil as ip
from dials.util.options import OptionParser, flatten_experiments
from dials.util.multi_dataset_handling import generate_experiment_identifiers
from dials.util.options import flatten_experiments

from dials.command_line.index import index
from dials.command_line.integrate import run_integration
from dials.command_line.refine import run_dials_refine

from dials.command_line import refine_bravais_settings as rbs
from cctbx import sgtbx, crystal
import json


class StrategyProcessor(object):
    """
    A processor variant designed to process two images taken at 90 degrees (or the
    like) for the purposes of strategy.
    """

    def __init__(self):
        pass

    def make_experiments(self, imagefiles):

        experiments = ExLF.from_filenames(filenames=imagefiles)

        # did input have identifier?
        had_identifiers = False
        if all(i != "" for i in experiments.identifiers()):
            had_identifiers = True
        else:
            generate_experiment_identifiers(
                experiments
            )  # add identifier e.g. if coming straight from images

        return experiments

    def find_spots(self, experiments, params):
        return flex.reflection_table.from_observations(experiments, params)

    def index(self, experiments, reflections, params):
        # Index
        experiments, indexed = index(experiments=experiments, reflections=[reflections],
                                     params=params)
        return indexed, experiments

    def refine_bravais_settings(self, experiments, indexed, params):
        crystals = experiments.crystals()

        # Determine SG
        reflections = rbs.eliminate_sys_absent(experiments, indexed)
        rbs.map_to_primitive(experiments, reflections)

        refined_settings = rbs.refined_settings_from_refined_triclinic(
            experiments, reflections, params
        )
        possible_bravais_settings = {solution["bravais"] for solution in
                                     refined_settings}
        rbs.bravais_lattice_to_space_group_table(possible_bravais_settings)

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

    def reindex(self, solution, reflections, experiments):
        # Reindex with new Bravais Lattice solution
        # Update space group / unit cell
        experiment = experiments[0]
        experiment.crystal.update(solution.refined_crystal)

        # Change basis
        cb_op = solution["cb_op_inp_best"].as_abc()
        change_of_basis_op = sgtbx.change_of_basis_op(cb_op)

        # Reindex experiments
        from dials.command_line.reindex import reindex_experiments

        experiments = reindex_experiments(
            experiments=experiments,
            cb_op=change_of_basis_op
        )

        # Reindex reflections
        miller_indices = reflections["miller_index"]
        non_integral_indices = change_of_basis_op.apply_results_in_non_integral_indices(
            miller_indices)
        sel = flex.bool(miller_indices.size(), True)
        sel.set_selected(non_integral_indices, False)
        miller_indices_reindexed = change_of_basis_op.apply(miller_indices.select(sel))
        reflections["miller_index"].set_selected(sel, miller_indices_reindexed)
        reflections["miller_index"].set_selected(~sel, (0, 0, 0))
        return experiments, reflections

    def refine(self, experiments, reflections, params):
        return run_dials_refine(params=params, reflections=reflections,
                                experiments=experiments)

    def process_reference(self, reference):
        """Load the reference spots."""
        if reference is None:
            return None, None
        assert "miller_index" in reference
        assert "id" in reference
        mask = reference.get_flags(reference.flags.indexed)
        rubbish = reference.select(~mask)
        if mask.count(False) > 0:
            reference.del_selected(~mask)
        if len(reference) == 0:
            raise Sorry(
                """
        Invalid input for reference reflections.
        Expected > %d indexed spots, got %d
      """
                % (0, len(reference))
            )
        mask = reference["miller_index"] == (0, 0, 0)
        if mask.count(True) > 0:
            rubbish.extend(reference.select(mask))
            reference.del_selected(mask)
        mask = reference["id"] < 0
        if mask.count(True) > 0:
            raise Sorry(
                """
        Invalid input for reference reflections.
        %d reference spots have an invalid experiment id
      """
                % mask.count(True)
            )
        return reference, rubbish

    def integrate_images(self, experiments, indexed, params):
        indexed, _ = self.process_reference(indexed)
        print('debug: {} indexed reflections remaining'.format(len(indexed)))

        # Get the integrator from the input parameters
        from dials.algorithms.integration.integrator import create_integrator
        from dials.algorithms.profile_model.factory import ProfileModelFactory

        # Compute the profile model
        # Predict the reflections
        # Match the predictions with the reference
        # Create the integrator
        experiments = ProfileModelFactory.create(params, experiments, indexed)
        predicted = flex.reflection_table.from_predictions_multi(
            experiments,
            dmin=params.prediction.d_min,
            dmax=params.prediction.d_max,
            margin=params.prediction.margin,
            force_static=params.prediction.force_static,
        )
        predicted.match_with_reference(indexed)
        integrator = create_integrator(params, experiments, predicted)

        # Integrate the reflections
        integrated = integrator.integrate()

        # Delete the shoeboxes used for intermediate calculations, if requested
        if params.integration.debug.delete_shoeboxes and "shoebox" in integrated:
            del integrated["shoebox"]

        return integrated, experiments

    def integrate(self, experiments, params, indexed=None):
        return self.integrate_images(experiments, indexed, params)

