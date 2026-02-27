from fltest.core import hooks

"""
class HookContext:

    cfg: Optional[Any] = None
    round: Optional[int] = None
    client_id: Optional[int] = None

    # Data phase
    raw_dataset: Optional[Any] = None
    partition_map: Optional[Dict[int, Any]] = None
    dist_dict: Optional[Dict[int, Any]] = None
    test_data: Optional[Any] = None

    # Model / state (Flower uses list of ndarrays for parameters)
    client_data: Optional[Any] = None  # per-client training data/loader
    global_state: Optional[Any] = None
    client_update: Optional[Any] = None
    num_samples: Optional[int] = None
    updates_and_weights: Optional[List[tuple]] = None  # list of (update, n)

    # Outputs
    new_global_state: Optional[Any] = None
    metrics: Optional[Dict[str, Any]] = None
    history: Optional[Dict[str, Any]] = field(default_factory=dict)
"""


@hooks.before_simulation
def log_before_simulation(ctx):
    pass


@hooks.on_data_distribute
def log_on_data_distribute(ctx):
    pass


@hooks.before_round
def log_before_round(ctx):
    pass


@hooks.before_client_train
def log_before_client_train(ctx):
    pass


@hooks.after_client_train
def log_after_client_train(ctx):
    pass


@hooks.after_round
def log_after_round(ctx):
    pass


@hooks.after_simulation
def log_after_simulation(ctx):
    pass
