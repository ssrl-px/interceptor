from dxtbx.model.experiment_list import ExperimentListFactory as ExLF
from dials.array_family import flex
from dials.util.multi_dataset_handling import generate_experiment_identifiers
from dials.command_line.index import index
from dials.command_line.refine import run_dials_refine
from dials.command_line.export_best import BestExporter

from dials.command_line import refine_bravais_settings as rbs
from cctbx import sgtbx, crystal
from libtbx.phil import Sorry

from iota.utils.utils import Capturing


class CustomBestExporter(BestExporter):
    def __init__(self, params, experiments, reflections):
        super().__init__(params, experiments, reflections)

    def export(self):
        """
        Export the files
        """
        from dials.util import best

        experiment = self.experiments[0]
        reflections = self.reflections[0]
        partiality = reflections["partiality"]
        sel = partiality >= self.params.min_partiality
        if sel.count(True) == 0:
            raise Sorry(
                "No reflections remaining after filtering for minimum partiality ("
                "min_partiality={.2f})".format(self.params.min_partiality)
            )
        reflections = reflections.select(sel)

        imageset = experiment.imageset
        prefix = self.params.output.prefix
        idx_prefix = "{}_".format(prefix)

        with Capturing() as junk_output:
            best.write_background_file(f"{prefix}.dat", imageset, n_bins=self.params.n_bins)
            best.write_integrated_hkl(idx_prefix, reflections)
            best.write_par_file(f"{prefix}.par", experiment)

        return f"{prefix}.dat", f"{prefix}.par", "{}_1.hkl".format(prefix), \
               "{}_2.hkl".format(prefix)


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

        # Pick out the highest-symmetry solution
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

        # Pick out the refined P1 solution
        P1_solution = [
            s for s in refined_settings if s["bravais"] == 'aP'
        ][0]

        return P1_solution, highest_sym_solution


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


def make_result_string(info, cfg):
    # collect errors
    err_list = [
        info[e] for e in info if ("error" in e or "comment" in e) and info[e] != ""
    ]
    errors = "; ".join(err_list)
    # Collect results
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

    # read out config format (if no path specified, read from default config file)
    if cfg.getstr('output_delimiter') is not None:
        delimiter = '{} '.format(cfg.getstr('output_delimiter'))
    else:
        delimiter = ' '
    format_keywords = cfg.getstr('output_format').split(',')
    format_keywords = [i.strip() for i in format_keywords]

    # assemble and return message to UI
    try:
        ui_msg = info[cfg.getstr('output_prefix_key')]
    except KeyError:
        ui_msg = ''
    if ui_msg == '':
        ui_msg = cfg.getstr('default_output_prefix')
    ui_msg += ' '
    for kw in format_keywords:
        keyword = kw
        bracket = None
        brackets = ['{}', '()', '[]']
        split_kw = kw.split(' ')
        if len(split_kw) == 2 and split_kw[1] in brackets:
            keyword = split_kw[0]
            bracket = split_kw[1]
        try:
            if kw.startswith('[') and kw.endswith(']'):
                keyword = ''
                value = info[kw[1:-1]]
            elif 'result' in keyword:
                value = results
            else:
                value = info[keyword]
        except KeyError as e:
            raise e
        else:
            if keyword == '':
                item = value
            elif bracket:
                item = '{0} {1}{2}{3}'.format(keyword, bracket[0],
                                              value, bracket[1])
            else:
                item = '{0} {1}'.format(keyword, value)
            if format_keywords.index(kw) == len(format_keywords) - 1:
                delimiter = ''
            ui_msg += item + delimiter
    return ui_msg


def print_to_stdout(counter, info, ui_msg, clip=False):
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
        ]
    except Exception as e:
        print(e)
    if clip:  # only print the worker output
        print(lines[1])
    else:
        for ln in lines:
            print(ln)
