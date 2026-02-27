from fltest.core import hooks

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
