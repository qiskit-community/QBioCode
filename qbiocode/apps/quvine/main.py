# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os 
import sys 
import logging 
from pathlib import Path 

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from qbiocode.apps.quvine.pipeline import Pipeline
from qbiocode.apps.quvine.utils.seed import set_global_seed 
from qbiocode.apps.quvine.utils.logging import setup_logging 
from qbiocode.apps.quvine.utils.io import save_config 

@hydra.main(
    
    config_path=None, 
    config_name=None,
    version_base="1.3"
    
)

def main(cfg: DictConfig) -> None: 
    """
    Hydra entrypoint for quvine experiments
    """
    
    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    
    run_dir = Path(os.getcwd()) #hydra changes cwd
    orig_cwd = Path(get_original_cwd()) 
    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    
    setup_logging(run_dir) 
    log = logging.getLogger(__name__)
    log.info("Starting quvine run")
    log.info("Run Directory: %s", run_dir)
    log.info("Project Root: %s", orig_cwd)
    
    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    
    if "seed" in cfg: 
        set_global_seed(cfg.seed)
        log.info("Global seed set to %d", cfg.seed)
        
    # ------------------------------------------------------------------
    # Resolved Config
    # ------------------------------------------------------------------
    
    save_config(cfg, run_dir/"config.yaml")
    log.debug("Resolved config: \n%s", OmegaConf.to_yaml(cfg))
    
    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    try:
        pipeline = Pipeline(cfg)
        pipeline.run()

    except Exception:
        # Bare `raise`, not `raise e`: re-raising the bound name appends this
        # frame to the traceback, so the reported origin is main() rather than
        # the line inside the pipeline that actually failed. log.exception has
        # already recorded the full traceback.
        log.exception("Experiment failed with error")
        raise

    log.info("Run completed successfully")
    
if __name__ == "__main__":
    main()
