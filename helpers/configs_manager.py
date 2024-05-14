import os
import sys
import json

from helpers import general_helpers as GH
class ConfigsManager:
    def __init__(self, file_path, lp, program_halt):
        self.lp = lp
        self.program_halt = program_halt
        self.configs = {}

        if not isinstance(file_path, str):
            self.program_halt("argument file_path should be string.")

        if not os.path.isfile(file_path):
            self.program_halt("config file not found: " + file_path)

        try:
            self.lp("reading configuration file: " + file_path)
            with open (file_path) as _file_handle:
                self.__configs_file_content = json.load (_file_handle)
            self.lp (["Config file abs-path: " + os.path.abspath(file_path) , "Config file contents:", "-" * 20, json.dumps (self.__configs_file_content, indent=4, sort_keys=True), "-" * 80])

        except Exception as E:
            self.program_halt("Error in reading config file: " + file_path + "\nError: " + str(E))

        self.configs['file_path'] = os.path.abspath(file_path)
        self.__process_configs()

    def __process_configs(self):
        c = self.__configs_file_content

        def config_error(msg):
            self.program_halt (["Error in processing config json file:"] + [msg])

        def config_relations_error(msg):
            err_msg = ["Error in processing config json file:"]
            err_msg+= [msg]
            err_msg+= ['example: "relations": {"Complex_formation": {"type": "undirectional" , "valid_pairs": ["Protein Protein" , "Protein Chemical" , "Protein Family", "Family Family"]}']
            self.program_halt(err_msg)

        #---------  classification_type: binary, multi-class, multi-label
        if not "classification_type" in c:
            config_error ("missing key: classification_type.")

        if not isinstance(c["classification_type"], str):
            config_error("value for classification_type in config file should be string")

        _classification_type = c["classification_type"].lower()
        if not _classification_type in ("binary" , "multi-class" , "multi-label"):
            config_error("value for classification_type in config file should be should be one of 'binary' , 'multi-class' or 'multi-label'.")
        self.configs['classification_type'] = _classification_type

        #---------  entities: what are valid entities.
        if not "entities" in c:
            config_error("missing key: entities.")

        if not isinstance(c["entities"], list):
            config_error("value for entities in config file should be a list.")

        if len(c["entities"]) < 1:
            config_error("entities in config file should be a list with at least one entity defined. Example: ['Protein'].")

        if len(set(c['entities'])) != len(c['entities']):
            config_error("error in entities section. Multiple definition of the same entity type?")

        self.configs["entities"] = c["entities"]

        #---------  relations --------:
        self.configs["relations"] = {}

        if not "relations" in c:
            config_error("missing key: relations.")

        if not isinstance(c["relations"], dict):
            config_relations_error("relations in config file should be a dictionary.")

        if len (c["relations"].keys()) < 1:
            config_relations_error("relations dictionary should contain at least one defined relation type.")

        if len(c["relations"].keys()) > 1 and (self.configs["classification_type"]=="binary"):
            config_error("more than one relation type are defined whereas the classification_type is defined as binary. It should be either multi-class or multi-label.")

        #TODO: fix later if you want 2 decision layers, one for relation_Type, one for direction
        if len(c["relations"].keys()) == 1 and (self.configs["classification_type"]=="binary"):
            only_rel_typle = list(c['relations'].keys())[0]
            try:
                if c["relations"][only_rel_typle]['type'].lower() == 'directional':
                    config_error("there is only one relation type, and it is DIRECTIONAL. Hence, you cannot define classification_type as binary.")
            except Exception as E:
                config_error("error in config file. propbably: there is only one relation type, and it is DIRECTIONAL. Hence, you cannot define classification_type as binary.")

        """
        #TODO: uncomment later. this is to allow test casting binary classification into multi-label and see what happens.
        if len(c["relations"].keys()) == 1 and (self.configs["classification_type"] in ("multi-class" , "multi-label")):
            only_rel_typle = list(c['relations'].keys())[0]
            try:
                if c["relations"][only_rel_typle]['type'].lower() == 'undirectional':
                    config_error("relations dictionary has only one UNDIRECTIONAL relation type whereas the classification_type is not defined as binary.")
            except Exception as E:
                config_error("error in config file. propbably: relations dictionary has only one UNDIRECTIONAL relation type whereas the classification_type is not defined as binary.")
        """
        if len(c["relations"].keys()) == 1 and (self.configs["classification_type"] in ("multi-class")):
            only_rel_typle = list(c['relations'].keys())[0]
            try:
                if c["relations"][only_rel_typle]['type'].lower() == 'undirectional':
                    config_error("relations dictionary has only one UNDIRECTIONAL relation type whereas the classification_type is not defined as binary.")
            except Exception as E:
                config_error("error in config file. propbably: relations dictionary has only one UNDIRECTIONAL relation type whereas the classification_type is not defined as binary.")


        for key in c["relations"].keys(): #check each relation type
            if not isinstance(c["relations"][key] , dict):
                config_relations_error("error in relations section.")

            if not "type" in c["relations"][key]:
                config_relations_error("missing 'type' in relations section.")

            if not c["relations"][key]["type"] in ("directional" , "undirectional"):
                config_relations_error("error in 'type' section of relations dictionary in relations section. It should be either directional or undirectional.")

            if not "valid_pairs" in c["relations"][key]:
                config_relations_error("missing 'valid_pairs' in relations section.")

            if not isinstance(c["relations"][key]["valid_pairs"], list):
                config_relations_error("valid_pairs in the relation section should be a list.")

            #<<<CRITICAL>>> VERY IMPORTANT NOTE:
            all_pairs = set()
            for items in  c["relations"][key]["valid_pairs"]:
                items = items.split(" ")
                if len(items) != 2:
                    config_relations_error("error in relations section.")
                e1,e2 = items
                if not e1 in self.configs["entities"]:
                    config_relations_error("undefined entity type : " + e1)
                if not e2 in self.configs["entities"]:
                    config_relations_error("undefined entity type : " + e2)

                if c["relations"][key]["type"] == "undirectional":
                    all_pairs.add((e1,e2))
                    all_pairs.add((e2,e1))
                elif c["relations"][key]["type"] == "directional":
                    all_pairs.add((e1,e2))

            self.configs["relations"][key] = {
                'type' : c["relations"][key]['type'],
                'valid_pairs': all_pairs
            }

        class RelationTypeEncoding:
            def __init__ (self, configs):
                if configs['classification_type'] in ["binary" , "multi-class"]:
                    counter = 1
                    relation_to_one_hot_index_mapping = {"neg": 0}
                elif configs['classification_type'] == "multi-label":
                    counter = 0
                    relation_to_one_hot_index_mapping = {}
                else:
                    print ("error in classification type in the config file. exiting ...")
                    sys.exit(-1)

                relation_type_to_writeback_relation_type_mapping = {} # for example: Complex_formation => Complex_formation, Regulation> => Regulation, Regulation< => Regulation, etc

                for relation_type in configs['relations'].keys():
                    if configs['relations'][relation_type]['type'] == "undirectional":
                        relation_to_one_hot_index_mapping[relation_type] = counter
                        relation_type_to_writeback_relation_type_mapping[relation_type] = relation_type
                        counter+=1
                    else:
                        rel_type_from_first_to_second_occurring = relation_type+">"
                        rel_type_from_second_to_first_occurring = relation_type+"<"

                        relation_to_one_hot_index_mapping[rel_type_from_first_to_second_occurring] = counter
                        relation_type_to_writeback_relation_type_mapping[rel_type_from_first_to_second_occurring] = relation_type
                        counter+=1

                        relation_to_one_hot_index_mapping[rel_type_from_second_to_first_occurring] = counter
                        relation_type_to_writeback_relation_type_mapping[rel_type_from_second_to_first_occurring] = relation_type
                        counter+=1

                one_hot_index_to_relation_mapping = {value:key for key, value in relation_to_one_hot_index_mapping.items()}
                self.relation_to_one_hot_index_mapping = relation_to_one_hot_index_mapping
                self.one_hot_index_to_relation_mapping = one_hot_index_to_relation_mapping
                self.relation_type_to_writeback_relation_type_mapping = relation_type_to_writeback_relation_type_mapping
                self.number_of_classes = len(self.relation_to_one_hot_index_mapping.keys())
        self.configs['RelationTypeEncoding'] = RelationTypeEncoding(self.configs)

        self.lp (str(self.configs))
