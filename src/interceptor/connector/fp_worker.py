from __future__ import division, absolute_import, print_function

import time

from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.indexer import Indexer
from dials.array_family import flex
from interceptor.connector.processor import InterceptorBaseProcessor, ImageScorer


class FastProcessor(InterceptorBaseProcessor):
    def __init__(
            self,
            run_mode='fast',
            configfile=None,
            test=False,
            verbose=False,
    ):
        self.verbose = verbose
        self.processing_mode = "integration"
        InterceptorBaseProcessor.__init__(self, run_mode=run_mode,
                                          configfile=configfile, test=test)

    def process(self, info, filename=None, data=None, detector=None):
        info["phil"] = self.dials_phil.as_str()

        # Make ExperimentList object
        experiments, e_time = self.make_experiments(filename=filename, data=data, detector=detector)
        print (f"MAKING EXPERIMENTLIST OBJECT TIME = {e_time:.4f} seconds")

        # Spotfinding
        try:
            spf_time = time.time()
            observed = self.processor.count_spots(experiments)
            print(f"SPOTFINDING COMPLETE! FOUND {len(observed)} spots")
            print(f'SPOTFINDING TIME = {time.time() - spf_time:.4f} seconds')
        except Exception as err:
            import traceback
            spf_tb = traceback.format_exc()
            info["spf_error"] = "SPF ERROR: {}".format(str(err))
            info['spf_error_tback'] = spf_tb
            return info
        else:
            if observed.size() >= self.cfg.getint('min_Bragg_peaks'):
                start = time.time()
                scorer = ImageScorer(experiments=experiments, observed=observed, config=self.cfg)
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
                print(f"SCORING TIME = {time.time() - start:.4f} seconds")
            else:
                info["n_spots"] = observed.size()

        # Doing it here because scoring can reject spots within ice rings, which can
        # drop the number below the minimal limit
        if info['n_spots'] < self.cfg.getint('min_Bragg_peaks'):
            info[
                "spf_error"] = "Too few ({}) spots found!".format(
                observed.size())
        if info["n_spots"] <= self.cfg.getint('min_Bragg_peaks'):
            return info
        else:
            exposure_time_cutoff = self.cfg.getfloat('exposure_time_cutoff')
            if exposure_time_cutoff > info['exposure_time']:
                return info

        # Indexing
        start_idx = time.time()
        params, self.dials_phil = self.generate_params()
        params.indexing.method = 'fft1d'
        params.indexing.stills.candidate_outlier_rejection = True
        params.indexing.stills.refine_all_candidates = True
        observed.centroid_px_to_mm(experiments)
        observed.map_centroids_to_reciprocal_space(experiments)
        try:
            idxr = Indexer.from_parameters(observed, experiments, params=params)
            idxr.index()
            indexed = idxr.refined_reflections
            experiments = idxr.refined_experiments
        except (AssertionError, DialsIndexError, Exception) as err:
            indexed = []
            print (err)
            pass

        print (f"INDEXING FINISHED! FOUND {len(indexed)} indexed reflections!")
        print (f"INDEXING TIME = {time.time() - start_idx:.4f} seconds")

        # Reindex if possible
        rdx_time = time.time()
        try:
            solution = self.processor.refine_bravais_settings(indexed, experiments)
            if solution is not None:
                experiments, indexed = self.processor.reindex(indexed, experiments, solution)
            else:
                info[
                    "rix_error"] = "reindex error: symmetry solution not found!"
            if len(indexed) == 0:
                info["idx_error"] = "index error: no indexed reflections!"
        except Exception as err:
            info["idx_error"] = "index error: {}".format(str(err))
            indexed = []
            print(f"reindex error: {str(err)}")
            #return info
        else:
            if indexed:
                lat = experiments[0].crystal.get_space_group().info()
                sg = str((lat)).replace(" ", "")
                unit_cell = experiments[0].crystal.get_unit_cell().parameters()
                uc = " ".join(["{:.2f}".format(i) for i in unit_cell])
                info["n_indexed"] = len(indexed)
                info["sg"] = sg
                info["uc"] = uc

        print(f"RE-INDEXING TIME = {time.time() - rdx_time:.4f} seconds")

        # **** Integration **** #
        int_time = time.time()

        # process reference
        indexed, _  = self.process_reference(reference=indexed)

        # create integrator
        start = time.time()
        integrator, predicted = self.create_integrator(experiments=experiments, indexed=indexed, params=params)
        print (f"*** DEBUG: integrator creation time = {time.time() - start:.05f} sec")
        integrated = predicted

        # Full integration, keeping here just in case
        # try:
        #     integrated = integrator.integrate()
        # except Exception as err:
        #     info["int_error"] = "integration error: {}".format(str(err))
        #     import traceback
        #     traceback.print_exc()
        #     integrated = []
        # else:
        #     info["int_error"] = '{} integrated'.format(integrated.size())

        # old slow integrator (i.e. complete from DIALS)
        # with Capturing() as int_output:
        #     try:
        #         experiments, indexed = self.processor.refine(experiments, indexed)
        #         integrated = self.processor.integrate(experiments, indexed)
        #     except Exception as err:
        #         info["int_error"] = "integration error: {}".format(str(err))
        #         import traceback
        #         traceback.print_exc()
        #         # return info
        #     else:
        #         info["int_error"] = '{} integrated'.format(integrated.size())
        print (f"INTEGRATION TIME = {time.time() - int_time:.4f} seconds")

        if integrated:
            info['n_integrated'] = integrated.size()
        return info

    def run(self, info, filename=None, data=None, detector=None):
        start = time.time()
        info = self.process(filename=filename, info=info, data=data, detector=detector)
        info['proc_time'] = time.time() - start
        return info

    def create_integrator(self, experiments, indexed, params):
        from dials.algorithms.integration.integrator import create_integrator
        from dials.algorithms.profile_model.factory import ProfileModelFactory
        from dxtbx.model.experiment_list import ExperimentList

        experiments = ProfileModelFactory.create(params, experiments, indexed)
        new_experiments = ExperimentList()
        new_reflections = flex.reflection_table()
        for expt_id, expt in enumerate(experiments):
            if (
                    params.profile.gaussian_rs.parameters.sigma_b_cutoff is None
                    or expt.profile.sigma_b()
                    < params.profile.gaussian_rs.parameters.sigma_b_cutoff
            ):
                refls = indexed.select(indexed["id"] == expt_id)
                refls["id"] = flex.int(len(refls), len(new_experiments))
                del refls.experiment_identifiers()[expt_id]
                refls.experiment_identifiers()[len(new_experiments)] = expt.identifier
                new_reflections.extend(refls)
                new_experiments.append(expt)
        experiments = new_experiments
        indexed = new_reflections

        if len(experiments) == 0:
            raise RuntimeError("No experiments after filtering by sigma_b")

        predicted = flex.reflection_table.from_predictions_multi(
            experiments,
            dmin=params.prediction.d_min,
            dmax=params.prediction.d_max,
            margin=params.prediction.margin,
            force_static=params.prediction.force_static,
        )
        predicted.match_with_reference(indexed)
        integrator = create_integrator(params, experiments, predicted)
        return integrator, predicted

    def process_reference(self, reference):
        """Load the reference spots."""
        if reference is None:
            return None, None
        st = time.time()
        assert "miller_index" in reference
        assert "id" in reference
        mask = reference.get_flags(reference.flags.indexed)
        rubbish = reference.select(~mask)
        if mask.count(False) > 0:
            reference.del_selected(~mask)
        if len(reference) == 0:
            print(f"Invalid input for reference reflections. Expected > {0} indexed spots, got {len(reference)}")
        mask = reference["miller_index"] == (0, 0, 0)
        if mask.count(True) > 0:
            rubbish.extend(reference.select(mask))
            reference.del_selected(mask)
        mask = reference["id"] < 0
        if mask.count(True) > 0:
            print(f"Invalid input for reference reflections. {mask.count(True)} reference spots have an invalid experiment id ")
        return reference, rubbish
