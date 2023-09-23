from utils import get_hyperparameter_combinations

def test_for_hparam_cominations_count():
    # a test case to check that all possible combinations of paramers are indeed generated
    gamma= [0.001, 0.01, 0.1, 1]
    C = [1, 10, 100, 1000]
    param_groups = {
        "gamma": gamma,
        "C": C
    }
    params_combinations = get_hyperparameter_combinations(param_groups)
    
    assert len(params_combinations) == len(gamma) * len(C)

def test_for_hparam_cominations_values():    
    gamma = [0.001, 0.01]
    C = [1]
    param_groups = {
        "gamma": gamma,
        "C": C
    }
    params_combinations = get_hyperparameter_combinations(param_groups)
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in params_combinations) and (expected_param_combo_2 in params_combinations)