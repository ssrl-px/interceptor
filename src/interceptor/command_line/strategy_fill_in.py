import argparse
import time

from iotbx import phil as ip
from cctbx import crystal, miller
from iotbx.reflection_file_reader import any_reflection_file

from libtbx.phil.command_line import argument_interpreter as ArgInterpreter
from libtbx.utils import Sorry

from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as spf_scope
from dials.command_line.index import phil_scope as idx_scope
from dials.command_line.refine_bravais_settings import phil_scope as brv_scope
from dials.command_line.integrate import phil_scope as int_scope
from dials.command_line.export_best import phil_scope as exp_scope

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
spf_work_phil = ip.parse(custom_spf_string)
spf_phil = spf_scope.fetch(source=spf_work_phil)

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
idx_work_phil = ip.parse(custom_idx_string)
idx_phil = idx_scope.fetch(source=idx_work_phil)

custom_brv_sting = """
crystal_id = 0
output {
  directory = "."
  log = None
  prefix = None
  }
"""

brv_work_phil = ip.parse(custom_brv_sting)
brv_phil = brv_scope.fetch(source=brv_work_phil)

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
int_work_phil = ip.parse(custom_int_string)
int_phil = int_scope.fetch(source=int_work_phil)

custom_exp_string = """
  n_bins = 100
  min_partiality = 0.1
  output {
    log = None
    prefix = best
  }
"""
exp_work_phil = ip.parse(custom_exp_string)
exp_phil = exp_scope.fetch(source=exp_work_phil)


def parse_command_args():
    """ Parses command line arguments """
    parser = argparse.ArgumentParser(
        prog="strategy_fill-in.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("Fill-in Strategy"),
        epilog=("\n{:-^70}\n".format("")),
    )
    parser.add_argument(
        "path",
        type=str,
        nargs=1,
        default=None,
        help="Path to a test image file",
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        nargs=1,
        default=None,
        help="CrystalServer project ID",
    )
    parser.add_argument(
        "--mtz", '-m',
        type=str,
        nargs=1,
        default=None,
        help="Path to a merged dataset in MTZ format"
    )
    parser.add_argument(
        "--phi_start", '-f',
        type=int,
        nargs=1,
        default=0,
        help="Starting PHI angle for strategy sweep (degrees)"
    )
    parser.add_argument(
        "--phi_end", '-e',
        type=int,
        nargs=1,
        default=180,
        help="Ending PHI angle for strategy sweep (degrees)"
    )
    parser.add_argument(
        "--wedge", '-w',
        type=int,
        nargs=1,
        default=1,
        help="Predicted data collection wedge for strategy sweep (degrees)"
    )
    parser.add_argument(
        "--phi_step", '-s',
        type=int,
        nargs=1,
        default=5,
        help="Step between PHI angles for strategy sweep (degrees)"
    )
    parser.add_argument(
        "--resolution", '-r',
        type=float,
        nargs=1,
        default=None,
        help="Resolution extent of predicted spots for the strategy sweep (Angstroms)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print output to stdout"
    )
    return parser


def insert_phil_args(phil, phil_args):
    bad_args = []
    argument_interpreter = ArgInterpreter(master_phil=phil)
    for arg in phil_args:
        try:
            command_line_params = argument_interpreter.process(arg=arg)
            phil = phil.fetch(sources=[command_line_params])

        except Sorry:
            bad_args.append(arg)

    return phil, bad_args


def fix_phils(phil_args):
    phils = {
        "spf_phil": spf_phil,
        "idx_phil": idx_phil,
        "brv_phil": brv_phil,
        "int_phil": int_phil,
        "exp_phil": exp_phil,
    }
    for pname, phil in phils.items():
        fixed_phil, phil_args = insert_phil_args(phil, phil_args)
        phils[pname] = phil.fetch(source=fixed_phil)
    return phils


class Script(object):
    def __init__(self, args, phils):
        self.args = args
        self.processor = StrategyProcessor()
        self.imagefiles = args.path
        self.project_id = args.project
        self.dataset_file = self.get_mtz(dataset_file=args.dataset_file)
        self.verbose = args.verbose
        self.create_params(phils)

    def create_params(self, phils):
        for pname, phil in phils.items():
            paramname = pname.split('_')[0] + "_params"
            params = phils[pname].extract()
            setattr(self, paramname, params)
            setattr(self, pname, phil)

    def get_mtz(self, dataset_file):
        if dataset_file is None:
            assert self.project_id   # will throw exception at all times at this point
            return None
        else:
            return dataset_file[0]

    def predict_and_merge(self, experiments, ref_marray, start_phi, wedge_size=1, target_res=2.5):
        # create predictions
        predicted_all = flex.reflection_table()
        for i_expt, expt in enumerate(experiments):
            predicted_all = flex.reflection_table()
            scan = expt.scan
            image_range = scan.get_image_range()
            oscillation = scan.get_oscillation()
            scan.set_image_range(
                (
                    image_range[0] - wedge_size,
                    image_range[1] + wedge_size,
                )
            )
            scan.set_oscillation(
                (
                    start_phi - wedge_size * oscillation[1],
                    oscillation[1],
                )
            )
            # Populate the reflection table with predictions
            predicted = flex.reflection_table.from_predictions(
                expt, force_static=False, dmin=target_res
            )
            predicted["id"] = flex.int(len(predicted), i_expt)
            predicted_all.extend(predicted)
        #Test w/ miller set
        hkl = predicted_all['miller_index']

        ms_pred = miller.set(
            crystal_symmetry=crystal.symmetry(
                space_group_symbol=str(ref_marray.crystal_symmetry().space_group_info()),
                unit_cell=ref_marray.unit_cell(),
            ),
            anomalous_flag=False,
            indices=hkl,
            ).unique_under_symmetry()
        ms_pred.map_to_asu()
        ref_mset = ref_marray.set()

        data_pred = flex.int(len(ms_pred.indices()), 1)
        data_ref  = flex.int(len(ref_mset.indices()), 10)
        array_pred = miller.array(miller_set=ms_pred, data=data_pred).map_to_asu().merge_equivalents().array()
        array_ref = miller.array(miller_set=ref_mset, data=data_ref).map_to_asu().merge_equivalents().array()

        # concatenate predicted w/ reference
        ma_conc = array_ref.concatenate(other=array_pred).unique_under_symmetry()
        ma_conc.merge_equivalents(incompatible_flags_replacement=0)
        return array_pred, array_ref, ma_conc

    def run(self):
        if self.verbose:
            print('Starting fill-in strategy with the following file(s):')
            for img in self.imagefiles:
                print (img)
        # make experiments
        experiments = self.processor.make_experiments(self.imagefiles)

        # find spots
        if self.verbose:
            print("Spotfinding...")
        observed = self.processor.find_spots(
            experiments=experiments,
            params=self.spf_params
        )
        if self.verbose:
            print("Spotfinding DONE: found {} spots total".format(observed.size()))

        # extract indexing solution from dataset file
        if self.verbose:
            print ("Extracting reference space group and unit cell from MTZ file...")

        ref_hklin = any_reflection_file(file_name=self.dataset_file)
        ref_marray = ref_hklin.as_miller_arrays()[0].map_to_asu()
        ref_unit_cell = ref_marray.unit_cell().parameters()
        ref_space_group = ref_marray.crystal_symmetry().space_group_info()

        if self.verbose:
            print(f"reference space group = {ref_space_group}")
            print(f"reference unit cell = {ref_unit_cell}")
            print ('\n')

        # index
        # apply reference parameters (because I want the initial indexing solution to match reference, if possible)
        self.idx_params.indexing.known_symmetry.space_group = ref_space_group
        self.idx_params.indexing.known_symmetry.unit_cell = ref_unit_cell

        if self.verbose:
            print("Indexing...")
        indexed, experiments = self.processor.index(
            experiments=experiments,
            reflections=observed,
            params=self.idx_params
        )

        crystals = experiments.crystals()
        if self.verbose:
            print("Initial indexing solution: ")
            print(f"space group = {crystals[0].get_space_group().info()}")
            print(f"unit cell = {crystals[0].get_unit_cell().parameters()}")
            print ('\n')


        # Set prediction parameters
        if self.args.resolution is not None:
            target_dmin = self.args.resolution if type(self.args.resolution) == float else self.args.resolution[0]
        else:
            target_dmin = indexed.as_miller_array(experiments[0]).d_min()
        start_phi = self.args.phi_start if type(self.args.phi_start) == int else self.args.phi_start[0]
        wedge_size = self.args.wedge if type(self.args.wedge) == int else self.args.wedge[0]
        phi_step = self.args.phi_step if type(self.args.phi_step) == int else self.args.phi_step[0]
        phi_end = self.args.phi_end if type(self.args.phi_end) == int else self.args.phi_end[0]

        print (f"Starting set completeness: {ref_marray.completeness()*100:.3f}% | wedge: {wedge_size} deg | target res: {target_dmin:.2f}A")
        best_phi = 0
        best_comp = ref_marray.completeness()*100

        import copy

        for phi in range(start_phi, phi_end, phi_step):
            start = time.time()
            _, _, combined_array = self.predict_and_merge(
                experiments=copy.deepcopy(experiments), ref_marray=ref_marray,
                start_phi=phi, wedge_size=wedge_size, target_res=target_dmin)
            current_comp = combined_array.completeness()*100
            print (f"Combined completeness for PHI = {phi}-{phi+wedge_size} deg: {current_comp:.3f}% ({time.time()-start:.3f} s)")

            # determine if best result
            if current_comp > best_comp:
                best_comp = current_comp
                best_phi = phi

        # output MTZs:
        predicted_array, reference_array, combined_array = self.predict_and_merge(
            experiments=copy.deepcopy(experiments), ref_marray=ref_marray,
            start_phi=best_phi, wedge_size=wedge_size, target_res=target_dmin)
        print (f"Writing predicted spots to MTZ...")
        predicted_array.write_mtz(file_name="predicted.mtz")
        print (f"Writing reference spots to MTZ...")
        reference_array.write_mtz(file_name='reference.mtz')
        print (f"Writing best result (PHI = {best_phi} deg) to MTZ...")
        combined_array.write_mtz(file_name="best_combined.mtz")


def entry_point():
    global_start = time.time()
    args, phil_args = parse_command_args().parse_known_args()
    phils = fix_phils(phil_args)
    script = Script(args=args, phils=phils)
    script.run()
    print (f"Total running time: {time.time()-global_start:0.2f} seconds")

if __name__ == "__main__":
    entry_point()


# --> end
