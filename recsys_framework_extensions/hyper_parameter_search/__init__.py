from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative

from recsys_framework_extensions.decorators import log_calling_args

run_hyper_parameter_search_collaborative = log_calling_args(
    runHyperparameterSearch_Collaborative
)
