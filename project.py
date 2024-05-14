import os
import sys

#Code to add the module paths to python
current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))
#-------------------------------------------------------------------------
from helpers import logger
from helpers import configs_manager
#from helpers import brat_json_converter
#from helpers import example_generation

class Project:
    def __init__(self, log_file_path, configs_file_path, allow_append_log_if_file_exists=False):
        self.__logger = logger.Logger(log_file_path, allow_append_log_if_file_exists)
        self.lp = self.__logger.lp
        self.configs = configs_manager.ConfigsManager(configs_file_path , self.lp , self.program_halt).configs
        #self.brat_json_converter = brat_json_converter.brat_json_Converter(self.lp, self.program_halt, self.configs)
        #self.example_generator = example_generation.example_generator(self.lp , self.program_halt, self.configs)

    def program_halt(self, message):
        self.__logger.lp_halt(message)
        self.exit()

    def exit(self):
        if self.__logger.is_open():
            self.lp ("EXITING PROGRAM...")
        self.__logger.close()
        sys.exit(-1)






