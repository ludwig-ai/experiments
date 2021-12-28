# experiments
Reproducible benchmark experiments for Ludwig

The experiments for automl are in the __automl__ directory.
* The __heuristics__ subdirectory contains subdirectories for each dataset used to run extensive hyperparameter searches from which to derive automl heuristics.
* The __validation__ subdirectory contains subdirectories for each dataset used to validate the derived heuristics.

Each __dataset__ subdirectory contains the following scripts and configuration files, as appropriate:
* __Training__
  * Simple train validation of __concat__ model type:
    * Script w/Configuration: **train_concat_sanity_laptop.py**, **config_concat_sanity_laptop.yaml**
  * Simple train validation of __tabnet__ model type:
    * Script w/Configuration: **train_tabnet_sanity_laptop.py**, **config_tabnet_sanity_laptop.yaml**
  * Simple train validation of __transformer__ model type:
    * Script w/Configuration: **train_transf_sanity_laptop.py**, **config_transf_sanity_laptop.yaml**
  * Train validation of __best tabnet__ model configuration found in heuristics search runs
    * Script w/Configuration: **train_tabnet_reference_laptop.py**, **config_tabnet_reference_laptop.yaml**
  * Train validation of best tabnet model configuration found in heuristics search runs using __updated__ automatic feature type selection (if impacted)
    * Script w/Configuration: **train_tabnet_reference_auto.py**, **config_tabnet_reference_auto.yaml**


* __AutoML__
  * Automatically generate configuration for hyperparameter search via create_auto_config API 
    * Script: **get_auto_train_config.py**
    * Output for original feature type selection code: **auto_config.json.orig**
    * Output for updated feature type selection code: **auto_config.json.update**
    * Output for updated feature type selection code + automl code w/heuristics: **auto_config.json.automl**
  * Automatically generate and run configuration for hyperparameter search via auto_train API w/1hr time limit
    * Script: **run_auto_train_1hr.py**
    * Output: **hyperopt_statistics.json.1hr**
  * Automatically generate and run configuration for hyperparameter search via auto_train API w/2hr time limit
    * Script: **run_auto_train_2hr.py**
    * Output: **hyperopt_statistics.json.2hr**

