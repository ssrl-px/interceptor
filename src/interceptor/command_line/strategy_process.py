import os
import sys
import argparse
import json

from iotbx import phil as ip
from dxtbx.model.experiment_list import ExperimentListFactory as ExLF
from cctbx import sgtbx, crystal

from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as spf_scope
from dials.command_line.index import phil_scope as idx_scope
from dials.command_line.refine_bravais_settings import phil_scope as brv_scope
from dials.command_line.refine import phil_scope as ref_scope
from dials.command_line.integrate import phil_scope as int_scope
from dials.command_line.export_best import phil_scope as exp_scope
from dials.command_line.index import index
from dials.command_line.integrate import run_integration
from dials.command_line.refine import run_dials_refine
from dials.command_line import refine_bravais_settings as rbs
from dials.util.options import  flatten_experiments
from dials.util.multi_dataset_handling import generate_experiment_identifiers

from interceptor.connector import CustomBestExporter
from interceptor.connector import StrategyProcessor

# Custom PHIL for spotfinding
custom_spf_string = """
output {
    strong_filename = None
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
"""
spf_phil = ip.parse(custom_spf_string)
custom_spf_params = spf_scope.fetch(source=spf_phil).extract()

# Custom PHIL for indexing
custom_idx_string = """
output {
  experiments_filename = None
  indexed_filename = None
  refined_experiments_filename = None
  integrated_experiments_filename = None
  integrated_filename = None
  profile_filename = None
  integration_pickle = None
}
indexing {
  refinement_protocol {
    d_min_start = 2.0
  }
}
"""
idx_phil = ip.parse(custom_idx_string)
custom_idx_params = idx_scope.fetch(source=idx_phil).extract()

custom_brv_sting = """
crystal_id = 0
output {
  directory = "."
  log = None
  prefix = None
  }
"""

brv_phil = ip.parse(custom_brv_sting)
custom_brv_params = brv_scope.fetch(source=brv_phil).extract()

custom_ref_string = """
  output {
    experiments = None
    reflections = None
    }

"""
ref_phil = ip.parse(custom_ref_string)
custom_ref_params = ref_scope.fetch(source=ref_phil).extract()

custom_int_string = """
output {
  experiments = None
  reflections = None
  phil = None
  log = None
  report = None
}
create_profile_model = True
integration {
  integrator = 2d
  profile.fitting = False
  background {
    algorithm = simple
    simple {
      outlier.algorithm = plane
      model.algorithm = linear2d
    }
  }
}
"""
int_phil = ip.parse(custom_int_string)
custom_int_params = int_scope.fetch(source=int_phil).extract()

custom_exp_string = """
  n_bins = 100
  min_partiality = 0.1
  output {
    log = None
    prefix = best
  }
"""
exp_phil = ip.parse(custom_exp_string)
custom_exp_params = exp_scope.fetch(source=exp_phil).extract()


def parse_command_args():
    """ Parses command line arguments (only options for now) """
    parser = argparse.ArgumentParser(
        prog="strategy_process.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("ZMQ Stream Connector"),
        epilog=("\n{:-^70}\n".format("")),
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="*",
        default=None,
        help="Path to data or file with IOTA parameters",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print output to stdout"
    )
    return parser

class Script(object):
    def __init__(self, args):
        self.processor = StrategyProcessor()
        self.imagefiles = args.path
        self.verbose = args.verbose

    def run(self):

        if self.verbose:
            print ('Starting strategy using {} and {}'.format(self.imagefiles[0],
                                                              self.imagefiles[1]))
        # make experiments
        experiments = self.processor.make_experiments(self.imagefiles)

        # find spots
        if self.verbose:
            print ("Spotfinding...")
        observed = self.processor.find_spots(
            experiments=experiments,
            params=custom_spf_params
        )
        if self.verbose:
            print ("Spotfinding DONE: found {} spots total".format(observed.size()))

        # Populate spotfinding results
        spotfinding_dict = {}
        for i, experiment in enumerate(experiments):
            refl = observed.select(observed["id"] == i)
            overloads = refl.select(refl.is_overloaded(experiments) == True)
            spf_dict = {
                'imagefile'         : experiment.imageset.paths()[0],
                'spotTotal'         : len(refl),
                'inResOverlSpots'   : len(overloads)
            }
            spotfinding_dict.update({i + 1: spf_dict})

        # index
        if self.verbose:
            print ("Indexing...")
        indexed, experiments = self.processor.index(
            experiments=experiments,
            reflections=observed,
            params=custom_idx_params
        )

        if self.verbose:
            crystals = experiments.crystals()
            print("Initial indexing solution: ", crystals[0].get_space_group().info(),
                  crystals[0].get_unit_cell())
            print("{} indexed reflections".format(len(indexed)))

        # refine Bravais settings
        if self.verbose:
            print ("Determining the likeliest space group...")
        P1_solution, highest_sym_solution = self.processor.refine_bravais_settings(
            experiments=experiments,
            indexed=indexed,
            params=custom_brv_params
        )
        # the below can be expanded to all acceptable solutions
        solutions = [P1_solution, highest_sym_solution]

        # Reindex and integrate for all Bravais solutions
        solutions_dict = {}
        for i, solution in enumerate(solutions):
            experiments, reindexed = self.processor.reindex(
                solution=solution,
                reflections=indexed,
                experiments=experiments
            )
            crystal = experiments.crystals()[0]
            if self.verbose:
                print(crystal, "\n")
                print ("{} reindexed reflections".format(len(reindexed)))

            # populate index / reindex results
            uc = crystal.get_unit_cell()
            idx_dict = {
                "id": solution.setting_number,
                "metricfit": solution['max_angular_difference'],
                "rmsd": solution.rmsd,
                "spots": solution.Nmatches,
                "crystalsystem": solution['system'],
                "lattice": solution['bravais'],
                "unitcell": uc.parameters(),
                "volume": uc.volume(),
            }

            # integrate
            if self.verbose:
                print ("Integrating...")
            integrated, experiments = self.processor.integrate(
                experiments=experiments,
                indexed=reindexed,
                params=custom_int_params
            )
            if self.verbose:
                print ("{} integrated reflections".format(len(integrated)))

            crystal = experiments.crystals()[0]

            # export BEST parameters
            if self.verbose:
                print('exporting results...')
            custom_exp_params.output.prefix = "best{}".format(i+1)
            exporter = CustomBestExporter(
                params=custom_exp_params,
                reflections=[integrated],
                experiments=experiments
            )

            datfile, parfile, hkl1, hkl2 = exporter.export()

            s_dict = {

                "matrix"            : crystal.get_A(),  # UB matrix
                "dataForBest"       : os.path.abspath(os.path.join(os.curdir, datfile)),
                "paramFileForBest"  : os.path.abspath(os.path.join(os.curdir, parfile)),
                "hkl1"              : os.path.abspath(os.path.join(os.curdir, hkl1)),
                "hkl2"              : os.path.abspath(os.path.join(os.curdir, hkl2)),
                "indexResult"       : idx_dict,
            }
            solutions_dict.update({solution.setting_number: s_dict})

        # collate other information
        crystal = experiments.crystals()[0]
        info_dict = {
            "imagefiles"    : self.imagefiles,
            "spotfinding"   : spotfinding_dict,
            "solutions"     : solutions_dict,
        }

        with open('info.json', "w") as jf:
            json.dump(info_dict, jf)

        if self.verbose:
            print ("\n*****")
            info_print = json.dumps(info_dict, indent=1)
            print(info_print)



def entry_point():
    args, phil_args = parse_command_args().parse_known_args()
    assert len(args.path) == 2
    script = Script(args=args)
    script.run()


if __name__ == "__main__":
    entry_point()

# --> end
