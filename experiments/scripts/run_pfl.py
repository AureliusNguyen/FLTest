# experiments/scripts/run_flare.py

import hydra
from fl_testing.federated.apple_pfl.server import simulate_fl



@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    simulate_fl(cfg)
    

    

if __name__ == "__main__":
    main()
