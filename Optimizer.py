import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..Misc import json_dict_type_correction

class Optimizer(object):
    """ Classe base para os otimizadores """

    def __init__(self, optimizerDict: Dict[str, Any]):
        """
        Construtor da classe Optimizer

        Parâmetros
        ----------
        optimizerDict : dict
            Dicionário que contém as informações do otimizador.
        """
        self.type = "Optimizer_Base"
        self.method = "None"
        self.optimizerDict = optimizerDict

        # Variáveis de callback para registrar dados durante a otimização
        self.num_calls = 0  # Quantas vezes a função de custo foi chamada
        self.callback_count = 0  # Número de vezes que o callback foi chamado (= contagem de iterações)
        self.list_calls_inp = []  # Input de todas as chamadas
        self.list_calls_res = []  # Resultado de todas as chamadas
        self.decreasing_list_calls_inp = []  # Inputs que resultaram em decréscimo
        self.decreasing_list_calls_res = []  # Resultados que indicaram decréscimo
        self.list_callback_inp = []  # Inputs apenas no callback (correspondem às iterações)
        self.list_callback_res = []  # Resultados apenas no callback

    def load_from_optimizer_dict(self):
        """ Carrega a configuração do otimizador a partir do dicionário """
        pass

    def write_optimization_results_to_csv(self, parameterNames: Optional[List[str]] = None):
        """ Escreve os resultados da otimização em um arquivo CSV """
        if parameterNames is None:
            parameterNames = []

        # Usando pathlib para definir o nome do arquivo
        file_path = Path(f"{self.type}_{self.method}_calibration_results.csv")

        # Usando context manager (with) para garantir o fechamento seguro do arquivo
        with open(file_path, "w", encoding="utf-8") as fid:
            # Cabeçalho dos parâmetros
            for i in range(len(self.list_callback_inp[0])):
                if not parameterNames:
                    fid.write(f"Parameter-{i},")
                else:
                    fid.write(f"{parameterNames[i]},")
            
            fid.write("Calibration-Error\n")

            # Escrevendo as iterações
            for i in range(len(self.list_callback_inp)):
                for j in range(len(self.list_callback_inp[0])):
                    fid.write(f"{self.list_callback_inp[i][j]:10.5e},")
                fid.write(f"{self.list_callback_res[i]:10.5e}\n")


class Optimizer_ScipyOptimizeLocal(Optimizer):
    """ Optimizer usando a classe scipy.optimize.local """

    def __init__(self, optimizerDict: Dict[str, Any]):
        super().__init__(optimizerDict)
        self.type = "ScipyOptimizeLocal"
        self.method = "L-BFGS-B"
        self.jac = None
        self.hess = None
        self.maxfun = 1000
        self.tol = 0.001
        self.maxiter = 2000
        self.disp = True
        self.options = {}

        self.load_from_optimizer_dict()

    def load_from_optimizer_dict(self):
        if "method" in self.optimizerDict:
            self.method = self.optimizerDict["method"]
        if "jac" in self.optimizerDict:
            self.jac = self.optimizerDict["jac"]
        if "hess" in self.optimizerDict:
            self.hess = self.optimizerDict["hess"]
        if "maxfun" in self.optimizerDict:
            self.maxfun = self.optimizerDict["maxfun"]
        if "maxiter" in self.optimizerDict:
            self.maxiter = self.optimizerDict["maxiter"]
        if "tol" in self.optimizerDict:
            self.tol = self.optimizerDict["tol"]
        if "disp" in self.optimizerDict:
            if str(self.optimizerDict["disp"]).lower() == "true":
                self.disp = True
            elif str(self.optimizerDict["disp"]).lower() == "false":
                self.disp = False
            else:
                raise ValueError("Optimizer's disp option should be either True or False.")
        if "options" in self.optimizerDict:
            self.options = self.optimizerDict["options"]
            json_dict_type_correction(self.options)


class Optimizer_ScipyOptimizeGlobal(Optimizer):
    """ Optimizer usando a classe scipy.optimize.global """

    def __init__(self, optimizerDict: Dict[str, Any]):
        super().__init__(optimizerDict)
        self.type = "ScipyOptimizeGlobal"
        self.method = "brute"
        self.options = {}

        self.load_from_optimizer_dict()

    def load_from_optimizer_dict(self):
        if "method" in self.optimizerDict:
            self.method = self.optimizerDict["method"]
        if "options" in self.optimizerDict:
            self.options = self.optimizerDict["options"]
            json_dict_type_correction(self.options)


class Optimizer_Enumerator(Optimizer):
    """ Optimizer enumerando combinações de parâmetros fornecidas pelo usuário """

    def __init__(self, optimizerDict: Dict[str, Any]):
        super().__init__(optimizerDict)
        self.type = "enumerator"
        self.method = "None"
        self.options = {}
        self.parameter_combinations = []
        
        self.best_combination_index = -1
        self.best_calibration_score = np.inf

        self.load_from_optimizer_dict()

    def load_from_optimizer_dict(self):
        for combination in self.optimizerDict.get('parameter_combinations', []):
            self.parameter_combinations.append(combination)

    def minimize(self, func_to_minimize, args=None, callback=None):
        if args is None or len(args) < 2:
            raise ValueError("Arguments materialID_list and materialName_list are required.")

        materialID_list, materialName_list = args[0], args[1]
        n_combinations = len(self.parameter_combinations)

        for idx, combination in enumerate(self.parameter_combinations):
            ManningNs = [None] * len(materialID_list)

            for matID, ManningN in combination.items():
                if int(matID) in materialID_list:
                    ManningNs[materialID_list.index(int(matID))] = ManningN
                else:
                    raise ValueError("The material ID in the enumerator does not match the list of calibration parameters.")

            if None in ManningNs:
                raise ValueError("Not all materials are specified in the enumerator.")

            print(f"Parameter combination #{idx + 1} / {n_combinations}")
            print(f"     Material names: {materialName_list}")
            print(f"     Material IDs:   {materialID_list}")
            print(f"     Manning Ns:     {ManningNs}")

            score = func_to_minimize(ManningNs, materialID_list, materialName_list)

            if idx == 0 or score < self.best_calibration_score:
                self.best_combination_index = idx
                self.best_calibration_score = score

            if callback:
                callback(ManningNs)

        print(f"The best calibration parameter combination is #{self.best_combination_index + 1}")
        print(f"Minimum calibration score: {self.best_calibration_score}")
        print("Best parameter values:")
        print(f"     Material names: {materialName_list}")
        print(f"     Material IDs:   {materialID_list}")
        print(f"     Manning Ns:     {self.parameter_combinations[self.best_combination_index]}")