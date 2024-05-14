import os
import json
import itertools

class example_generator:
    def __init__(self, lp, program_halt, configs):
        self.lp = lp #lp: function that prints to output and also log file, you can pass print as well.
        self.program_halt = program_halt # the function that prints and also into the log file, and then run sys.exit(-1)
        self.configs = configs
        self.__validate_configs()

    def __validate_configs(self):
        #very basic checks
        if not isinstance(self.configs , dict):
            self.program_halt("error in configs. not a dictionary")
        if not "entities" in self.configs:
            self.program_halt("entities key not found in configs")
        if not "relations" in self.configs:
            self.program_halt("relations key not found in configs")
        if not isinstance(self.configs['relations'], dict):
            self.program_halt("relations in the configs should be a dictionary")

    def check_file_exists(self, file_address):
        if not os.path.isfile(file_address):
            self.program_halt("invalid file address: " + file_address)

    def generate_examples (self, input_json_fileaddress, output_json_fileaddress = None, input_json_conent=None, dont_generate_negatives_if_sentence_distance_ge=None):
        global_counter_interactions = 0
        counter_annotated_positives = 0
        counter_generated_positives = 0
        counter_generated_negatives = 0

        #check function arguments
        if isinstance(input_json_fileaddress, str):
            if len(input_json_fileaddress) == 0:
                input_json_fileaddress = None

        if (input_json_fileaddress is None) and (input_json_conent is None):
            self.program_halt("both of input_json_fileaddress and input_json_conent are None. EXACTLY one of them should be given.")

        if (input_json_fileaddress is not None) and (input_json_conent is not None):
            self.program_halt("both of input_json_fileaddress and input_json_conent are NOT None. EXACTLY one of them should be given.")

        if (dont_generate_negatives_if_sentence_distance_ge is not None):
            if not isinstance(dont_generate_negatives_if_sentence_distance_ge, int):
                self.program_halt("dont_generate_negatives_if_sentence_distance_ge should be an integer, and >=1")
            if dont_generate_negatives_if_sentence_distance_ge < 1:
                self.program_halt("dont_generate_negatives_if_sentence_distance_ge should be an integer, and >=1")

        #Create a dictionary for DIRECTED pair_types to valid relations based on the config file.
        #This takes into account (Chem, Prot) and (Prot, Chem) ARE different.
        #Since in config_manager, we have already created symmetric pair types for undirectional relation_types (such as Complex_formation),
        #we don't face any problems here.
        # For example, for Complex_formation, if only (Prot, Family) existed in the config file, but not (Family, Prot),
        # we have added (Family, Prot) to valid_pair types in the config_manager code.
        # For directional relation types, we rely on what is defined in the config file, i.e., we don't generate the symmetric pair, becuase this
        # gives more control for different tasks.
        directed_pair_types_to_valid_relations_mapping = {}
        for relation_type in self.configs['relations'].keys():
            for e1TP_e2TP in self.configs['relations'][relation_type]['valid_pairs']:
                if not e1TP_e2TP in directed_pair_types_to_valid_relations_mapping:
                    directed_pair_types_to_valid_relations_mapping[e1TP_e2TP] = set()
                directed_pair_types_to_valid_relations_mapping[e1TP_e2TP].add(relation_type)

        #check and read the input file or use the given json_content
        if input_json_fileaddress is not None:
            self.check_file_exists(input_json_fileaddress)
            try:
                with open(input_json_fileaddress, "r", encoding='utf-8') as jsonfile:
                    json_data = json.load(jsonfile)
            except Exception as E:
                self.program_halt("Error reading file.\nError:" + str(E))
        else:
            json_data = input_json_conent

        #process the json input file
        for document in json_data['documents']:
            # 1) reading annotated equivs
            # make a dictionary for fast accessing all equivalents of entities ...
            # example:
            #    - input list of list: [['T10', 'T11', 'T12', 'T13']]
            #    - outpu dictionary: --> {'T10': {'T12', 'T13', 'T11'}, 'T11': {'T10', 'T12', 'T13'}, 'T12': {'T10', 'T13', 'T11'}, 'T13': {'T10', 'T12', 'T11'}}

            # TODO: for now, we assume all equivs have the same entity type
            annotated_equivs = document['equivs']
            all_equivs_dict = {}
            for equiv_set in annotated_equivs:
                for entity_id in equiv_set:
                    if not entity_id in all_equivs_dict:
                        all_equivs_dict[entity_id] = set(equiv_set) - set([entity_id])
                    else:
                        all_equivs_dict[entity_id] |= set(equiv_set) - set([entity_id])

            #2) read annotated inner + cross_sentence entities and relations
            #all_valid_entities= {key:value for key, value in document['cross_sentence_entities'].items() if value['tag'] in self.configs['entities']}
            all_valid_entities = {key: value for key, value in document['entities'].items() if value['tag'] in self.configs['entities']}

            # getting relations without paying attention to their arguments type, but only based on verifying the relation_type to be valid.
            #all_possibly_valid_relations = {key:value for key, value in document['cross_sentence_relations'].items() if value['type'] in self.configs['relations'].keys()}
            all_possibly_valid_relations = {key:value for key, value in document['relations'].items() if value['type'] in self.configs['relations'].keys()}

            """
            #for previous json file format that corresponding entities and relations existed and had been assigned to the sentences. (cross_sentences were on top of the document). 
            #3) read annotated entities and relations in the sentences and ADD them to the corresponding dictionaries.
            for sentence in document['sentences']:
                # getting entities
                this_sentence_valid_annotated_enities = {key:value for key, value in sentence['entities'].items() if value['tag'] in self.configs['entities']}
                # getting relations without paying attention to their arguments type, but only based on checking the relation type.
                this_sentence_possibly_valid_annotated_relations = {key:value for key, value in sentence['relations'].items() if value['type'] in self.configs['relations'].keys()}

                #add this sentence entities to the dictionary of all entities
                for key, value in this_sentence_valid_annotated_enities.items():
                    if not key in all_valid_entities:
                        all_valid_entities[key] = value
                    else:
                        self.program_halt('error in processing input json: duplicate entitiy:\nkey: ' + str(key) + "\nvalue: "+str(value))

                #add this sentence possibly valid relations to the dictionary of all possibly valid relations
                for key, value in this_sentence_possibly_valid_annotated_relations.items():
                    if not key in all_possibly_valid_relations:
                        all_possibly_valid_relations[key] = value
                    else:
                        self.program_halt('error in processing input json: duplicate relation:\nkey: ' + str(key) + "\nvalue: "+str(value))
            """

            #4) generate all interactions (inner-sentence and cross-sentence)
            document['pairs'] = []
            positives_intermediate = dict() #key will be frozenset([e1,e2]), and values will be the list of all relations for this pair
            all_negatives = set()

            #4-1) processing POSSIBLY valid positives:
            # - skipping (removing) invalid relations.
            # - also, aggregating all relevant relations into dict, while PRESERVING the correct direction of the two entities.
            # - also, taking into account equivs of e1, and equivs of e2 into account, while preserving the original order of e1 and e2.
            # - example:
            #    Input
            #        R1: Complex_formation(e1,e2)
            #        R2: Positive_regulation(e1,e2)
            #        R3: Negative_regulation(e2,e1)
            #   Output:
            #        all_positives[frozenset([e1,e2])] = [(R1, Complex_formation(e1,e2)) , (R2, Positive_regulation(e1,e2)), (R3, Negative_regulation(e2,e1))]
            #
            #   Note: frozenset(['e1','e2']) == frozenset(['e2','e1']) --> True
            #
            #   Then, after aggregation, we look if e1 is occurring first in the document or e2 and then we will either of these:
            #   if e1_orig_offset_bgn < e2_orig_offset_bgn:
            #       [e1,e2] = [Complex_formation, Positive_regulation> , Negative_regulation<]
            #   else:
            #       [e2,e1] = [Complex_formation, Positive_regulation< , Negative_regulation>]

            for relation_id in all_possibly_valid_relations:
                relation_type = all_possibly_valid_relations[relation_id]['type']
                e1_id, e2_id = [x[1] for x in all_possibly_valid_relations[relation_id]['arguments']]

                #Skipping invalid relation
                #Check1: skipping those pairs which are not valid according to config.
                #   For example, Complex_formation(Protein1 , Family1) might be valid in general, but in our particular config file, maybe 'Family' is not defined as a valid entity type
                #   in the entities section. Therefore, no entities with type == 'Family' are added into all_valid_entities.
                if (not e1_id in all_valid_entities) or (not e2_id in all_valid_entities):
                    continue

                #Check2: skipping if (e1_tp, e2_tp) is not valid in general, i.e., not defined for <<<ANYYYYY>>> relation_type in the config file.
                #    Note: no problem for undirectional relation_types like complex_formation(Prot,Family), because if complex_formation(Prot,Family) is defined in the file,
                #          but NOT complex_formation(Family,Prot), then config_manager script, automatically has added complex_formation(Family,Prot) into the list of valid things.
                e1_tp, e2_tp = all_valid_entities[e1_id]['tag'], all_valid_entities[e2_id]['tag']
                if not (e1_tp, e2_tp) in directed_pair_types_to_valid_relations_mapping.keys():
                    continue

                #Check3: skipping if (e1_tp, e2_tp) is not valid for THIS PARTICULAR relation_type.
                # For example (Prot, Prot) maybe valid for complex_formation, but not for a particular relation type.
                if not relation_type in directed_pair_types_to_valid_relations_mapping[(e1_tp, e2_tp)]:
                    continue

                counter_annotated_positives += 1

                #Get all equivs of e1 and all equivs of e2, while checking if each equiv is valid type.
                e1_equivs = {e1_id}
                e2_equivs = {e2_id}
                if e1_id in all_equivs_dict:
                    e1_equivs |= set([entity_id for entity_id in all_equivs_dict[e1_id] if entity_id in all_valid_entities])

                if e2_id in all_equivs_dict:
                    e2_equivs |= set([entity_id for entity_id in all_equivs_dict[e2_id] if entity_id in all_valid_entities])

                #try to sort equiv sets beautifully ...
                try:
                    e1_equivs = sorted(e1_equivs, key=lambda x:int(x[1:]))
                except:
                    e1_equivs = sorted(e1_equivs)

                try:
                    e2_equivs = sorted(e2_equivs, key=lambda x:int(x[1:]))
                except:
                    e2_equivs = sorted(e2_equivs)

                #generating positives for the original AND the equivs, WHILE aggregating separate relations into frozensets, WHILE preserving e1, e2 order.
                for e1_eqiuvalent in e1_equivs: #first loop on e1_equivs to keep the original order
                    for e2_equivalent in e2_equivs:#second loop on e2_qquivs to keep the original order
                        if not frozenset([e1_eqiuvalent, e2_equivalent]) in positives_intermediate:
                            positives_intermediate[frozenset([e1_eqiuvalent, e2_equivalent])] = []
                        positives_intermediate[frozenset([e1_eqiuvalent, e2_equivalent])].append(
                            {
                                "Orig_Relation_Id": relation_id,
                                "Arg1": e1_eqiuvalent, #preserving order
                                "Arg2": e2_equivalent, #preserving order
                                "type": relation_type
                            }
                        )
                        counter_generated_positives += 1

            #4-2) AGGREGATE individual positives based on the frozenset([e1,e2])
            #        positives_intermediate[frozenset([e1,e2])] --> a list of annotated relations
            # AND Generate Single annotation per list with all labels.
            for pair in positives_intermediate.keys():
                orig_pair = pair #keep frozenset for later iteration
                pair = list(pair)
                e_a , e_b = pair[0] , pair[1]
                is_inner_sentence = (len(all_valid_entities[e_a]['sentence_id']) == 1) and \
                                    (len(all_valid_entities[e_b]['sentence_id']) == 1) and \
                                    (all_valid_entities[e_a]['sentence_id'] == all_valid_entities[e_b]['sentence_id'])
                e_a_bgn_offset = all_valid_entities[e_a]['orig_spans'][0][0]
                e_b_bgn_offset = all_valid_entities[e_b]['orig_spans'][0][0]

                if e_a_bgn_offset < e_b_bgn_offset:
                    first_occurring_entity  = e_a
                    second_occurring_entity = e_b
                else:
                    first_occurring_entity  = e_b
                    second_occurring_entity = e_a

                example_info = {
                    "id": "i" + str(global_counter_interactions),
                    "e1": first_occurring_entity,
                    "e2": second_occurring_entity,
                    "labels": None,
                    "cross_sentence" : not (is_inner_sentence)
                }

                found_labels = set()

                for rel_info in positives_intermediate[orig_pair]:
                    Arg1 = rel_info['Arg1']
                    #Arg2 = rel_info['Arg2']
                    orig_rel_type = rel_info['type']

                    if self.configs["relations"][orig_rel_type]['type'] == 'undirectional':
                        found_labels.add(orig_rel_type)
                    else:
                        #look which has happened first.
                        if Arg1 == first_occurring_entity:
                            found_labels.add(orig_rel_type+">")
                        else:
                            found_labels.add(orig_rel_type + "<")

                example_info["labels"] = sorted(found_labels)
                document["pairs"].append(example_info)
                global_counter_interactions += 1

            #4-3) Generate negatives
            #first check if we should not generate examples if the two entities are X sentences apart
            can_process_sentence_distance = True
            if dont_generate_negatives_if_sentence_distance_ge is not None:
                for e_id in all_valid_entities.keys():
                    try:
                        sen_id = int(all_valid_entities[e_id]['sentence_id'][0][1:])
                        all_valid_entities[e_id]['sentence_int_id'] = sen_id
                    except:
                        can_process_sentence_distance= False
                        break

            for candidate_pair in itertools.combinations(all_valid_entities.keys(), 2):
                _e1_tp = all_valid_entities[candidate_pair[0]]['tag']
                _e2_tp = all_valid_entities[candidate_pair[1]]['tag']

                if (dont_generate_negatives_if_sentence_distance_ge is not None) and (can_process_sentence_distance):
                    _e1_s_id = all_valid_entities[candidate_pair[0]]['sentence_int_id']
                    _e2_s_id = all_valid_entities[candidate_pair[1]]['sentence_int_id']
                    abs_e1_e2_s_distance = abs(_e1_s_id - _e2_s_id)
                    if abs_e1_e2_s_distance >= dont_generate_negatives_if_sentence_distance_ge:
                        continue

                x = frozenset(candidate_pair)
                if (not x in positives_intermediate) and ((_e1_tp, _e2_tp) in directed_pair_types_to_valid_relations_mapping.keys()):
                    # the second condition is important because for example 'chemical chemical' may have not been defined in the config file as <<<ANY>>> valid pair type.
                    # note: order of (e1,e2) does not matter since all orders for directional and undirectional relation types have been taken care of in configs_manager.
                    all_negatives.add(x)
                    counter_generated_negatives += 1

            all_negatives_sorted = [list(x) for x in all_negatives]
            try:
                all_negatives_sorted = [sorted(list(x), key=lambda i: int(i[1:])) for x in all_negatives_sorted]
                all_negatives_sorted = sorted(all_negatives_sorted, key=lambda x: (int(x[0][1:]), int(x[1][1:])))
            except Exception as E:
                all_negatives_sorted = sorted(list(i) for i in all_negatives_sorted)

            for pair in all_negatives_sorted:
                e_a , e_b = pair[0] , pair[1]
                is_inner_sentence = (len(all_valid_entities[e_a]['sentence_id']) == 1) and \
                                    (len(all_valid_entities[e_b]['sentence_id']) == 1) and \
                                    (all_valid_entities[e_a]['sentence_id'] == all_valid_entities[e_b]['sentence_id'])
                e_a_bgn_offset = all_valid_entities[e_a]['orig_spans'][0][0]
                e_b_bgn_offset = all_valid_entities[e_b]['orig_spans'][0][0]

                if e_a_bgn_offset < e_b_bgn_offset:
                    first_occurring_entity  = e_a
                    second_occurring_entity = e_b
                else:
                    first_occurring_entity  = e_b
                    second_occurring_entity = e_a

                example_info = {
                    "id": "i" + str(global_counter_interactions),
                    "e1": first_occurring_entity,
                    "e2": second_occurring_entity,
                    "labels": [],
                    "cross_sentence": not (is_inner_sentence)
                }
                document["pairs"].append(example_info)
                global_counter_interactions += 1

        if output_json_fileaddress is not None:
            try:
                with open(output_json_fileaddress, "w", encoding='utf-8') as jsonfile:
                    json.dump(json_data, jsonfile, ensure_ascii=False)
                    #self.lp("output: " + output_json_fileaddress)
            except Exception as E:
                self.program_halt("cannot create json file. Error: " + str(E))

        return json_data, counter_annotated_positives, counter_generated_positives, counter_generated_negatives
