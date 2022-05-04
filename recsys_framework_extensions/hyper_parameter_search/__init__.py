import abc

from recsys_framework_extensions.data.io import DataIO
from HyperparameterTuning.SearchAbstractClass import SearchAbstractClass as RecSysFrameworkSearchAbstractClass
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt as RecSysFrameworkSearchBayesianSkopt
from HyperparameterTuning.SearchSingleCase import SearchSingleCase as RecSysFrameworkSearchSingleCase
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative

from recsys_framework_extensions.decorators import log_calling_args

run_hyper_parameter_search_collaborative = log_calling_args(
    runHyperparameterSearch_Collaborative
)


class SearchAbstractClass(RecSysFrameworkSearchAbstractClass, abc.ABC):
    def _set_search_attributes(
        self,
        recommender_input_args,
        recommender_input_args_last_test,
        hyperparameter_names,
        metric_to_optimize,
        cutoff_to_optimize,
        output_folder_path,
        output_file_name_root,
        resume_from_saved,
        save_metadata,
        save_model,
        evaluate_on_test,
        n_cases,
        terminate_on_memory_error
    ):
        super()._set_search_attributes(
            recommender_input_args=recommender_input_args,
            recommender_input_args_last_test=recommender_input_args_last_test,
            hyperparameter_names=hyperparameter_names,
            metric_to_optimize=metric_to_optimize,
            cutoff_to_optimize=cutoff_to_optimize,
            output_folder_path=output_folder_path,
            output_file_name_root=output_file_name_root,
            resume_from_saved=resume_from_saved,
            save_metadata=save_metadata,
            save_model=save_model,
            evaluate_on_test=evaluate_on_test,
            n_cases=n_cases,
            terminate_on_memory_error=terminate_on_memory_error,
        )

        if self.save_metadata:
            self.dataIO = DataIO(folder_path=self.output_folder_path)


class SearchSingleCase(SearchAbstractClass, RecSysFrameworkSearchSingleCase):
    pass


class SearchBayesianSkopt(SearchAbstractClass, RecSysFrameworkSearchBayesianSkopt):
    pass


