"""
Parametric Study.

Sampling in the parameter space.
"""
import json
import copy
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

# --- Importações atualizadas com o nome HydroSim ---
from HydroSim.Calibration.Parameters import Parameters
from HydroSim.Parametric_Study.Sampler import Sampler_SkoptSampler, Sampler_Enumerator
import HydroSim

from skopt.space import Space
from ..__common__ import gVerbose

class Parametric_Study:
    """ Base class para estudos paramétricos """

    def __init__(self, config_json_file_name: str):
        self.config_json_file_name = config_json_file_name
        self.model_name = ""
        self.sampler_name = ""
        self.configuration: Dict[str, Any] = {}
        
        self.load_config_json()
        
        self.hydraulic_model = None
        self.hydraulic_data = None
        self.create_model_data()

        self.parameters = Parameters(self.configuration["parametric_study"]["parameters"])
        self.n_samples = self.configuration["parametric_study"]["n_samples"]
        
        self.sampler = None
        self.create_sampler()
        
        self.logger = logging.getLogger(__name__)
        self.init_logger()

    def load_config_json(self):
        if gVerbose:
            print(f"Load parametric study configuration from file: {self.config_json_file_name}")

        with open(self.config_json_file_name, "r", encoding="utf-8") as f:
            self.configuration = json.load(f)

        self.model_name = self.configuration.get("model", "")
        self.sampler_name = self.configuration["parametric_study"]["sampler"]

        if gVerbose:
            print(f"Model = {self.model_name}")
            print(f"Configuration for {self.model_name}:")
            print(json.dumps(self.configuration.get(self.model_name, {}), indent=4))
            print(f"There are total {len(self.configuration['parametric_study']['parameters'])} parameters:")
            print(json.dumps(self.configuration["parametric_study"]["parameters"], indent=4))
            print(f"The selected sampler is {self.sampler_name} with the following setup:")
            print(json.dumps(self.configuration["parametric_study"][self.sampler_name], indent=4))

    def create_model_data(self):
        if self.model_name == "Backwater-1D":
            self.hydraulic_model = HydroSim.Hydraulic_Models_Data.Backwater_1D_Model()
            self.hydraulic_data = HydroSim.Hydraulic_Models_Data.Backwater_1D_Data(self.config_json_file_name)
            self.hydraulic_model.set_simulation_case(self.hydraulic_data)

        elif self.model_name == "SRH-2D":
            srh_config = self.configuration["SRH-2D"]
            self.hydraulic_model = HydroSim.SRH_2D.SRH_2D_Model(
                srh_config["version"], 
                srh_config["srh_pre_path"],
                srh_config["srh_path"], 
                srh_config.get("extra_dll_path", ""), 
                faceless=False
            )
            self.hydraulic_model.init_model()

            if gVerbose:
                print(f"Hydraulic model name: {self.hydraulic_model.getName()}")
                print(f"Hydraulic model version: {self.hydraulic_model.getVersion()}")

            self.hydraulic_model.open_project(srh_config["case"])
            self.hydraulic_data = self.hydraulic_model.get_simulation_case()

        elif self.model_name == "HEC-RAS":
            ras_config = self.configuration["HEC-RAS"]
            faceless = str(ras_config.get("faceless", "False")).lower() == "true"
            
            self.hydraulic_model = HydroSim.RAS_2D.HEC_RAS_Model(ras_config["version"], faceless)
            self.hydraulic_model.init_model()

            print(f"Hydraulic model name: {self.hydraulic_model.getName()}")
            print(f"Hydraulic model version: {self.hydraulic_model.getVersion()}")
            
        else:
            raise ValueError(f"The specified model '{self.model_name}' is not supported.")

    def create_sampler(self):
        if self.sampler_name == "skopt.sampler":
            self.sampler = Sampler_SkoptSampler(self.configuration["parametric_study"]["skopt.sampler"])
        elif self.sampler_name == "enumerator":
            self.sampler = Sampler_Enumerator(self.configuration["parametric_study"]["enumerator"])
        else:
            raise ValueError(f"Specified sampler '{self.sampler_name}' is not supported.")

    def init_logger(self):
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('parametric_study.log', mode='w')

        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        self.logger.setLevel(logging.INFO)

    def create_all_cases(self):
        print("Creating all cases ...")

        cases_dir = Path("cases")
        if cases_dir.is_dir():
            raise FileExistsError(f"The directory '{cases_dir}' already exists. Please remove it before running.")
        cases_dir.mkdir()

        param_space_list = self.parameters.get_parameter_space()
        param_space = Space(param_space_list)
        samples = self.sampler.generate(param_space, self.n_samples)
        
        print(f"Samples: {samples}")

        if self.model_name == "SRH-2D":
            for i, sample in enumerate(samples):
                print(f"Current sample [{i}]: {sample}")
                sample_np = np.array(sample)

                my_srh_2d_srhhydro = copy.deepcopy(self.hydraulic_data.srhhydro_obj)
                srh_caseName = f"{my_srh_2d_srhhydro.get_Case_Name()}_{i}"

                # Gerenciamento de arquivos usando Path
                newGridFileName = f"{srh_caseName}.srhgeom"
                newHydroMatFileName = f"{srh_caseName}.srhmat"
                newSRHHydroFileName = f"{srh_caseName}.srhhydro"

                oldGridFileName = my_srh_2d_srhhydro.get_Grid_FileName()
                oldHydroMatFileName = my_srh_2d_srhhydro.get_HydroMat_FileName()

                shutil.copy(oldGridFileName, newGridFileName)
                shutil.copy(oldHydroMatFileName, newHydroMatFileName)

                my_srh_2d_srhhydro.modify_Case_Name(srh_caseName)
                my_srh_2d_srhhydro.modify_Grid_FileName(newGridFileName)
                my_srh_2d_srhhydro.modify_HydroMat_FileName(newHydroMatFileName)

                for p_idx, parameter in enumerate(self.parameters.parameter_list):
                    if parameter.type == "ManningN":
                        my_srh_2d_srhhydro.modify_ManningsN(
                            [parameter.materialID], [sample_np[p_idx]], [parameter.name]
                        )
                    elif parameter.type == "InletQ":
                        my_srh_2d_srhhydro.modify_InletQ(
                            [parameter.bcID], [sample_np[p_idx]]
                        )

                my_srh_2d_srhhydro.write_to_file(newSRHHydroFileName)

                case_folder = cases_dir / f"case_{i}"
                case_folder.mkdir()

                shutil.move(newSRHHydroFileName, case_folder)
                shutil.move(newHydroMatFileName, case_folder)
                shutil.move(newGridFileName, case_folder)

        elif self.model_name == "HEC-RAS":
            for i, sample in enumerate(samples):
                print(f"Current sample [{i}]: {sample}")
                sample_np = np.array(sample)
                # Lógica HEC-RAS pendente
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented for parametric study.")

        return samples, self.parameters.get_parameter_name_list()

    def close_and_cleanup(self):
        if self.model_name == "HEC-RAS":
            self.hydraulic_model.exit_model()