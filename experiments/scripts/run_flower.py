# experiments/scripts/run_flare.py

import hydra
from fl_testing.federated.flower.server import run_flower_simulation



@hydra.main(config_path="config", config_name="config")
def main(cfg):
    run_flower_simulation(cfg)
    

    

if __name__ == "__main__":
    main()
