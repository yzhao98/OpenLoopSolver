from manipulator.data.constants import ReachingSetup
from manipulator.data.init_sampler_ee import sample_initial_states_based_on_ee
from manipulator.data.init_sampler_impls import (
    sample_initial_states_cube,
    sample_initial_states_simple,
)
from manipulator.data.utils import (
    get_initial_terminal_generalized_coordinates,
)
from manipulator.data.gen_ddp_solution import construct_solver
