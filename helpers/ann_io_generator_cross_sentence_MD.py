import os
import sys
import json
import math
import numpy as np
from collections import OrderedDict

#Code to add the module paths to python
current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))

from helpers import pipeline_variables

class ANN_IO_Generator(object):
    def __init__(self, lp, program_halt, configs, BERT_helper) :
        self.lp = lp
        self.program_halt = program_halt
        self.configs = configs
        self.BERT_helper = BERT_helper
        self.__create_model_markers_mapping()

    def __create_model_markers_mapping(self):
        counter = 3 #let's reserve [unused1] for masking focus entities and [unused2] for masking non-relevant (other) entities.
        markers_mapping = {}
        for entity_type in self.configs['entities']:
            markers_mapping[(entity_type, 0)] = self.BERT_helper.vectorize_without_taking_max_len_into_account(['[unused' + str(counter) + ']'])
            markers_mapping[(entity_type, 1)] = self.BERT_helper.vectorize_without_taking_max_len_into_account(['[unused' + str(counter + 1) + ']'])
            counter+=2
        markers_mapping['[CLS]'] = [self.BERT_helper.tokenizer.cls_token_id] #get correct CLS token_id based on whatever is used in this particular model (because CLS token might be something like '<s>' and not '[CLS]'
        markers_mapping['[SEP]'] = [self.BERT_helper.tokenizer.sep_token_id] #get correct SEP token_id based on whatever is used in this particular model (because SEP token might be something like </s>' and not '[SEP]'
        markers_mapping['[PAD]'] = [self.BERT_helper.tokenizer.pad_token_id] #get correct PAD token_id based on whatever is used in this particular model (because PAD token might be something like <pad>' and not '[PAD]'
        markers_mapping['MASK_focus_entity'] = ['[unused1]']
        markers_mapping['MASK_other_entity'] = ['[unused2]']
        self.model_markers_mapping_to_ids = markers_mapping

    def read_json_file(self, json_file_path):
        if not os.path.isfile(json_file_path):
            self.program_halt("invalid file address: " + str(json_file_path))
        try:
            with open(json_file_path, "r", encoding='utf-8') as jsonfile:
                json_data = json.load(jsonfile)
                return json_data
        except Exception as E:
            err_msg = "Error loading json file: " + json_file_path + "\nError: " + str(E)
            self.program_halt(err_msg)

    def model_tokenize_calculate_all_entity_boundaries_MARK(self, document_element):
        all_valid_entities = {key:value for key,value in document_element['entities'].items() if value['tag'] in self.configs['entities']}

        #add entity_id also as a field_value to each item. Will be to be used later.
        for entity_id in all_valid_entities:
            all_valid_entities[entity_id]['entity_id'] = entity_id

        #check if we don't have multi-span entities
        for key, value in all_valid_entities.items():
            if len(value['orig_spans']) > 1:
                err_msg = "multi-span entities are not supported:\n"
                err_msg+= "document_id: " + document_element['id'] + "\n"
                err_msg+= "entity     : " + str(value)
                self.program_halt(err_msg)

        document_raw_text = document_element['text']
        sorted_entities = sorted([value for key,value in all_valid_entities.items()], key= lambda x:x['orig_spans'][0][0])

        raw_text_last_offset = 0
        document_tokens = []

        for entity_info in sorted_entities:
            entity_id  = entity_info['entity_id']
            entity_bgn = entity_info['orig_spans'][0][0]
            entity_end = entity_info['orig_spans'][0][1]
            text_before = document_raw_text[raw_text_last_offset:entity_bgn]
            if len(text_before)>0:
                document_tokens.extend(self.BERT_helper.tokenize_text(text_before))
            entity_bert_tokens = self.BERT_helper.tokenize_text(document_raw_text[entity_bgn:entity_end])
            entity_bert_tokens_bgn_offset = len(document_tokens)
            entity_bert_tokens_end_offset = entity_bert_tokens_bgn_offset + len(entity_bert_tokens)
            document_tokens.extend(entity_bert_tokens)
            all_valid_entities[entity_id]['bert_tokens_offsets'] = (entity_bert_tokens_bgn_offset, entity_bert_tokens_end_offset)
            raw_text_last_offset = entity_end

        #add the remaining part of the sentence ...
        text_after = document_raw_text[raw_text_last_offset:]
        if len(text_after)>0:
            document_tokens.extend(self.BERT_helper.tokenize_text(text_after))

        #vectorize bert_tokens WITHOUT taking into account bert max_seq_len , not adding [CLS] or [SEP]
        document_tokens_indices = self.BERT_helper.vectorize_without_taking_max_len_into_account(document_tokens)
        return all_valid_entities, document_tokens, document_tokens_indices

    def model_tokenize_calculate_all_entity_boundaries_MASK_EVERYTHING(self, document_element):
        #first, add INNER + cross-sentence entities if there are any
        all_valid_entities = {key:value for key,value in document_element['entities'].items() if value['tag'] in self.configs['entities']}

        """
        #uncomment for previous file format in which inner-sentence entities, were added to their corresponding sentences.  
        #now add inner-sentence entities, sentence by sentence
        for sentence in document_element['sentences']:
            for entity_id in sentence['entities']:
                if sentence['entities'][entity_id]['tag'] in self.configs['entities']:
                    all_valid_entities[entity_id] = sentence['entities'][entity_id]
        """

        #add entity_id also as a field_value to each item, to be used later.
        for entity_id in all_valid_entities:
            all_valid_entities[entity_id]['entity_id'] = entity_id

        #check if we don't have multi-span entities
        for key, value in all_valid_entities.items():
            if len(value['orig_spans']) > 1:
                err_msg = "multi-span entities are not supported:\n"
                err_msg+= "document_id: " + document_element['id'] + "\n"
                err_msg+= "entity     : " + str(value)
                self.program_halt(err_msg)

        document_raw_text = document_element['text']
        sorted_entities = sorted([value for key,value in all_valid_entities.items()], key= lambda x:x['orig_spans'][0][0])

        raw_text_last_offset = 0
        document_tokens = []
        document_tokens_indices = []
        document_tokens_offsets = []

        for entity_info in sorted_entities:
            entity_id  = entity_info['entity_id']
            entity_bgn = entity_info['orig_spans'][0][0]
            entity_end = entity_info['orig_spans'][0][1]
            #ADD text_before ei
            text_before = document_raw_text[raw_text_last_offset:entity_bgn]
            if len(text_before)>0:
                bert_tokens = self.BERT_helper.tokenize_text(text_before)
                document_tokens.extend(bert_tokens)
                document_tokens_indices.extend(self.BERT_helper.vectorize_without_taking_max_len_into_account(bert_tokens))
                tokens_relative_offsets = self.BERT_helper.tokenizer(text_before , add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'] #token offsets in text_before
                tokens_orig_offsets = [(i[0]+raw_text_last_offset, i[1]+raw_text_last_offset) for i in tokens_relative_offsets] #token offsets in the whole document
                document_tokens_offsets.extend(tokens_orig_offsets)#extend

            #Add ei itself
            document_tokens.append([entity_id]) #deliberately, add entity_id as a list, to seprate it from ordinary bert tokens.
            document_tokens_indices.append([entity_id]) #deliberately, add entity_id as a list, to seprate it from integer numbers.
            all_valid_entities[entity_id]['entity_index'] = len(document_tokens_indices) - 1 # where the entity_id is located in the vectorized thingy.
            raw_text_last_offset = entity_end
            document_tokens_offsets.extend([(entity_bgn,entity_end)])

        #ADD the remaining part of the document after the LAST entity ...
        text_after = document_raw_text[raw_text_last_offset:]
        if len(text_after)>0:
            bert_tokens = self.BERT_helper.tokenize_text(text_after)
            document_tokens.extend(bert_tokens)
            document_tokens_indices.extend(self.BERT_helper.vectorize_without_taking_max_len_into_account(bert_tokens))
            tokens_relative_offsets = self.BERT_helper.tokenizer(text_after , add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'] #token offsets in text_before
            tokens_orig_offsets = [(i[0]+raw_text_last_offset, i[1]+raw_text_last_offset) for i in tokens_relative_offsets] #token offsets in the whole document
            document_tokens_offsets.extend(tokens_orig_offsets)#extend

        return all_valid_entities, document_tokens, document_tokens_indices, document_tokens_offsets #all withouot CLS/SEP , also no PAD

    def __generate_single_pair_BERT_MARK(self, document_tokens_indices, all_valid_entities, example):
        #get entity ids
        e1_info = all_valid_entities[example['e1']]
        e2_info = all_valid_entities[example['e2']]

        #find which occurs first
        if e1_info['orig_spans'][0][0] < e2_info['orig_spans'][0][0]:
            first_occurring_entity = e1_info
            secnd_occurring_entity = e2_info
        else:
            first_occurring_entity = e2_info
            secnd_occurring_entity = e1_info

        #distance calculation
        bert_seq_max_len = self.BERT_helper.MAX_POSSIBLE_SEQUENCE_LENGTH
        e1_e2_between_distance_in_tokens = secnd_occurring_entity['bert_tokens_offsets'][1] - first_occurring_entity['bert_tokens_offsets'][0]
        e1_length_in_tokens = first_occurring_entity['bert_tokens_offsets'][1] - first_occurring_entity['bert_tokens_offsets'][0]
        e2_length_in_tokens = secnd_occurring_entity['bert_tokens_offsets'][1] - secnd_occurring_entity['bert_tokens_offsets'][0]
        e1_e2_total_distance = e1_length_in_tokens + e2_length_in_tokens + e1_e2_between_distance_in_tokens

        #we should reserve the following 6 tokens: [CLS] , [SEP] , e1_bgn_marker, e1_end_marker, e2_bgn_marker, e2_end_marker
        #now let's check if we can afford generating this example
        if (e1_e2_total_distance + 6) > bert_seq_max_len:
            return None, e1_e2_between_distance_in_tokens

        #now let's calculate how many tokens before the first and after the second occurring entities we can afford
        count_tokens_around = math.floor((bert_seq_max_len - (e1_e2_between_distance_in_tokens + 6)) / 2.0)

        #index before first occurring entity
        if (first_occurring_entity['bert_tokens_offsets'][0] - count_tokens_around) > 0:
            index_before = first_occurring_entity['bert_tokens_offsets'][0] - count_tokens_around
        else:
            index_before = 0

        #index after second occurring entity
        index_last = len(document_tokens_indices) - 1
        if (secnd_occurring_entity['bert_tokens_offsets'][1] + count_tokens_around) < index_last:
            index_after = secnd_occurring_entity['bert_tokens_offsets'][1] + count_tokens_around
        else:
            index_after = index_last

        this_example_tokens_indices = []
        this_example_tokens_indices+= self.model_markers_mapping_to_ids['[CLS]']
        this_example_tokens_indices+= document_tokens_indices[index_before:first_occurring_entity['bert_tokens_offsets'][0]] #add text before
        this_example_tokens_indices+= self.model_markers_mapping_to_ids[(first_occurring_entity['tag'], 0)] #bgn
        this_example_tokens_indices+= document_tokens_indices[first_occurring_entity['bert_tokens_offsets'][0]:first_occurring_entity['bert_tokens_offsets'][1]]
        this_example_tokens_indices+= self.model_markers_mapping_to_ids[(first_occurring_entity['tag'], 1)] #end
        this_example_tokens_indices+= document_tokens_indices[first_occurring_entity['bert_tokens_offsets'][1]:secnd_occurring_entity['bert_tokens_offsets'][0]]
        this_example_tokens_indices+= self.model_markers_mapping_to_ids[(secnd_occurring_entity['tag'], 0)] #bgn
        this_example_tokens_indices+= document_tokens_indices[secnd_occurring_entity['bert_tokens_offsets'][0]:secnd_occurring_entity['bert_tokens_offsets'][1]]
        this_example_tokens_indices+= self.model_markers_mapping_to_ids[(secnd_occurring_entity['tag'], 1)] #end
        this_example_tokens_indices+= document_tokens_indices[secnd_occurring_entity['bert_tokens_offsets'][1]:index_after]
        this_example_tokens_indices+= self.model_markers_mapping_to_ids['[SEP]']

        #print ("-"*80)
        #print (first_occurring_entity['text'])
        #print (secnd_occurring_entity['text'])
        #print (example['labels'] , label)
        #print (len(this_example_bert_tokens_vectorized))
        #print (this_example_bert_tokens_vectorized)

        return this_example_tokens_indices, e1_e2_between_distance_in_tokens

    def __generate_single_pair_BERT_MASK_EVERYTHING(self, document_tokens_indices, document_tokens_offsets, all_valid_entities, example):
        #example:
            # documents_tokens       : ['and' , ['entity_id_1'] , 'binds' , 'to' , ['entity_id_2'] , '.'] #does not include CLS/SEP
            # document_tokens_indices: [ 123  , ['entity_id_1'] , 124325  , 234  , ['entity_id_2'] , 364] #does not include CLS/SEP
            # document_tokens_offsets: a list of pairs [(0,4), (5,7), etc]  #does not include CLS/SEP

        #get entity ids
        e1_info = all_valid_entities[example['e1']]
        e2_info = all_valid_entities[example['e2']]

        #find which occurs first
        if e1_info['orig_spans'][0][0] < e2_info['orig_spans'][0][0]:
            first_occurring_entity = e1_info
            secnd_occurring_entity = e2_info
        else:
            first_occurring_entity = e2_info
            secnd_occurring_entity = e1_info

        #distance calculation
        bert_seq_max_len = self.BERT_helper.MAX_POSSIBLE_SEQUENCE_LENGTH
        e1_e2_between_distance_in_tokens = secnd_occurring_entity['entity_index'] - first_occurring_entity['entity_index']

        #we should reserve the following tokens: [CLS] , [SEP] , e1_mask token, e2_mask_token
        #now let's check if we can afford generating this example
        if (e1_e2_between_distance_in_tokens + 4) > bert_seq_max_len:
            return None, None, e1_e2_between_distance_in_tokens

        #now let's calculate hoow many tokens before the first and after the second occurring entities we can afford
        count_tokens_around = math.floor((bert_seq_max_len - (e1_e2_between_distance_in_tokens + 4)) / 2.0)

        #token index before first occurring
        if (first_occurring_entity['entity_index'] - count_tokens_around) > 0:
            index_before = first_occurring_entity['entity_index'] - count_tokens_around
        else:
            index_before = 0

        #token index after second occurring
        index_last = len(document_tokens_indices) - 1
        if (secnd_occurring_entity['entity_index'] + count_tokens_around) < index_last:
            index_after = secnd_occurring_entity['entity_index'] + count_tokens_around
        else:
            index_after = index_last

        #init important variables
        this_example_tokens_indices = [] #will hold the final tokens' indices which will go to ANN as input
        this_example_tokens_offsets = [] #will hold offsets of the tokens ...
        focus_entity_token = self.model_markers_mapping_to_ids['MASK_focus_entity']
        focus_entity_token_index = self.BERT_helper.vectorize_without_taking_max_len_into_account(focus_entity_token)[0]
        other_entity_token = self.model_markers_mapping_to_ids['MASK_other_entity']
        other_entity_token_index =  self.BERT_helper.vectorize_without_taking_max_len_into_account(other_entity_token)[0]

        #add CLS
        this_example_tokens_indices+= self.model_markers_mapping_to_ids['[CLS]']
        this_example_tokens_offsets+= [(0,0)] #according to documentation, offset of tokens which don't appear in text is zero.

        #add before
        this_example_tokens_indices+= [i if isinstance(i, int) else other_entity_token_index for i in document_tokens_indices[index_before:first_occurring_entity['entity_index']]]

        #mask e1 with FOCUS_ENTITY token
        this_example_tokens_indices+= [focus_entity_token_index]

        #between
        this_example_tokens_indices+= [i if isinstance(i, int) else other_entity_token_index for i in document_tokens_indices[first_occurring_entity['entity_index'] + 1:secnd_occurring_entity['entity_index']]]

        #mask e2 FOCUS_ENTITY token
        this_example_tokens_indices+= [focus_entity_token_index]

        #After
        this_example_tokens_indices += [i if isinstance(i, int) else other_entity_token_index for i in document_tokens_indices[secnd_occurring_entity['entity_index'] + 1:index_after]]

        #[SEP]
        this_example_tokens_indices+= self.model_markers_mapping_to_ids['[SEP]']

        this_example_tokens_offsets.extend(document_tokens_offsets[index_before:index_after])
        this_example_tokens_offsets.extend([(0,0)]) #for SEP

        return this_example_tokens_indices, this_example_tokens_offsets, e1_e2_between_distance_in_tokens

    def __encode_example_label(self, example):
        decision_layer_dim = self.configs['RelationTypeEncoding'].number_of_classes
        label = np.zeros((decision_layer_dim), dtype=np.int8)
        # <<<CRITICAL>>> we don't generate label['neg'] = 1 for multi-label, because there is no 'neg' dimension in multi-label. i.e., if negative, then all dimensions are zero.
        # but for binary and multi-class, THERE IS a 'neg' dimension in the output, hence it should be set to 1 if example is negative...
        if example['labels'] == []: #if negative ...
            if self.configs['classification_type'] in ["binary", "multi-class"]:
                label[self.configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['neg']] = 1
        else: # if positive ...
            for _lbl_name in example['labels']:
                label[self.configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping[_lbl_name]] = 1
        return label

    def generate_ANN_Input_Outputs_pairs(self, json_data, generate_output=True, strategy = pipeline_variables.BERT_Representation_Strategy.MARK_FOCUS_ENTITIES , print_results=False, return_for_pt=False):
        if strategy == pipeline_variables.BERT_Representation_Strategy.MASK_FOCUS_ENTITIES:
            self.program_halt("strategy not implemented yet. ")
            #todo: implement later if needed.

        pair_tracking, ANN_inputs , ANN_outputs = [] , [] , []
        all_labels = []
        all_examples_tokens_indices = []
        all_e1_e2_between_distances_in_tokens = []

        for document in json_data['documents']:
            if strategy == pipeline_variables.BERT_Representation_Strategy.MARK_FOCUS_ENTITIES:
                all_valid_entities, document_tokens, document_tokens_indices = self.model_tokenize_calculate_all_entity_boundaries_MARK (document)
            elif strategy == pipeline_variables.BERT_Representation_Strategy.MASK_EVERYTHING:
                all_valid_entities, document_tokens, document_tokens_indices, document_tokens_offsets = self.model_tokenize_calculate_all_entity_boundaries_MASK_EVERYTHING(document)

            #print(all_valid_entities)
            #print(document_bert_tokens)
            #print(document_tokens_indices)

            document_id = document['id']
            for example in document["pairs"]:
                example_id = example['id']
                example_e1_id = example['e1'] #first entity
                example_e2_id = example['e2'] #second entity

                if generate_output:
                    label = self.__encode_example_label(example)
                else:
                    label = None

                if strategy == pipeline_variables.BERT_Representation_Strategy.MARK_FOCUS_ENTITIES:
                    this_example_tokens_indices, e1_e2_distance = self.__generate_single_pair_BERT_MARK(document_tokens_indices, all_valid_entities, example)
                    this_example_tokens_offsets = None
                elif strategy == pipeline_variables.BERT_Representation_Strategy.MASK_EVERYTHING:
                    this_example_tokens_indices, this_example_tokens_offsets, e1_e2_distance = self.__generate_single_pair_BERT_MASK_EVERYTHING(document_tokens_indices, document_tokens_offsets, all_valid_entities, example)

                all_e1_e2_between_distances_in_tokens.append(e1_e2_distance)

                #TODO: gather statistics about these discarded guys ?
                #<<<CRITICAL>>> SKIP IF CANNOT GENERATE EXAMPLE !!!
                if this_example_tokens_indices is None:
                    continue

                """
                print ("-"*80)
                print(example)
                print(all_valid_entities[example['e1']])
                print(all_valid_entities[example['e2']])
                for a in zip(this_example_tokens_indices, this_example_tokens_offsets):
                    idx, bgn, end = a[0], a[1][0], a[1][1]
                    print(idx, (bgn, end), pt_helper.tokenizer.convert_ids_to_tokens([idx]), [document['text'][bgn:end]])
                print ("-"*80)
                """

                pair = (document_id, example_id , example_e1_id, example_e2_id, this_example_tokens_indices , this_example_tokens_offsets)
                pair_tracking.append(pair)  # list of tuples
                all_examples_tokens_indices.append(this_example_tokens_indices)  # list of list ...
                if generate_output:
                    all_labels.append(label)  # list of list

        number_of_examples = len(pair_tracking)
        if number_of_examples < 1: #<<<CRITICAL>>>
            return None, None, None

        bert_matrix = np.ones((number_of_examples, self.BERT_helper.MAX_POSSIBLE_SEQUENCE_LENGTH), dtype=np.int32) * self.model_markers_mapping_to_ids['[PAD]'][0]
        for i in range(number_of_examples):
            bert_matrix[i, 0:len(all_examples_tokens_indices[i])] = all_examples_tokens_indices[i]
        ANN_inputs.append(bert_matrix)
        ANN_inputs.append(np.zeros_like(bert_matrix, dtype=np.int8))  # second always all-zero bert input
        if generate_output:
            ANN_outputs.append(np.vstack(all_labels))

        if return_for_pt:
            if not generate_output:
                pt_examples_labels = None
            else:
                if self.configs['classification_type'] in ["binary", "multi-class"]:
                    pt_examples_labels = [np.argmax(item) for item in all_labels]
                else:
                    pt_examples_labels = all_labels

            return {"pair_tracking": pair_tracking ,
                    "examples_input_ids" : all_examples_tokens_indices ,
                    "examples_labels" : pt_examples_labels}
        else:
            return pair_tracking , ANN_inputs , ANN_outputs


    def generate_ANN_Inputs_Outputs_with_FalseNegatives(self, json_data, generate_output: bool, strategy: pipeline_variables.BERT_Representation_Strategy):
        """
        Since there is a max_seq_len, not all training/dev/test pairs can be turned into machine learning examples.
        In the case of devel/test, those examples should be considered as false-negatives.
        This version of function returns necessary info about those examples.

        Some candidate-pairs:
        - will fit into max_seq_len, so they will be turned into actual machine learning examples (train/dev/test)
        - will NOT fit into max_seq_len, AND IF THEY ARE DEV/TEST examples, they should be considered as False-Negativess of the system in score calculation.
        """
        if strategy == pipeline_variables.BERT_Representation_Strategy.MASK_FOCUS_ENTITIES:
            self.program_halt("strategy not implemented yet. ")
            #todo: implement later if needed.

        fitted_pairs_tracking, fitted_examples_tokens_indices, ANN_inputs , ANN_outputs = [] , [] , [] , []
        all_e1_e2_between_distances_in_tokens = []

        # For gatgering info about labels (whether generate_output is True or False).
        # 1. define needed variables
        all_labels_statistics_for_report_dict = OrderedDict()
        fitted_labels_list_of_npvectors , unfitted_labels_list_of_npvectors = [] , []

        # 2. make variables suitable
        all_task_labels = set(self.configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping.keys())
        if self.configs['classification_type'] == 'multi-label':
            all_task_labels.add('neg') #for the sake of making reports.
        for _lbl_name in ['neg'] + sorted(all_task_labels - {'neg'}):
            all_labels_statistics_for_report_dict[_lbl_name] = dict()
            all_labels_statistics_for_report_dict[_lbl_name]['fitted'] = 0
            all_labels_statistics_for_report_dict[_lbl_name]['unfitted'] = 0

        # start processing documents in json, one-by-one ...
        for document in json_data['documents']:
            if strategy == pipeline_variables.BERT_Representation_Strategy.MARK_FOCUS_ENTITIES:
                all_valid_entities, document_tokens, document_tokens_indices = self.model_tokenize_calculate_all_entity_boundaries_MARK (document)
            elif strategy == pipeline_variables.BERT_Representation_Strategy.MASK_EVERYTHING:
                all_valid_entities, document_tokens, document_tokens_indices, document_tokens_offsets = self.model_tokenize_calculate_all_entity_boundaries_MASK_EVERYTHING(document)

            document_id = document['id']
            for example in document["pairs"]:
                example_id = example['id']
                example_e1_id = example['e1'] #first entity
                example_e2_id = example['e2'] #second entity

                # make example's ANN input (X)
                if strategy == pipeline_variables.BERT_Representation_Strategy.MARK_FOCUS_ENTITIES:
                    this_example_tokens_indices, e1_e2_distance = self.__generate_single_pair_BERT_MARK(document_tokens_indices, all_valid_entities, example)
                    this_example_tokens_offsets = None
                elif strategy == pipeline_variables.BERT_Representation_Strategy.MASK_EVERYTHING:
                    this_example_tokens_indices, this_example_tokens_offsets, e1_e2_distance = self.__generate_single_pair_BERT_MASK_EVERYTHING(document_tokens_indices, document_tokens_offsets, all_valid_entities, example)

                all_e1_e2_between_distances_in_tokens.append(e1_e2_distance)

                # make example's ANN output (Y)
                if generate_output:
                    example_labels_list = example['labels']  # a list, can be negative like [] , or can contain one label like ['Complex_formation'], or more labels like: ['Catalysis_of_dephosphorylation<', 'Regulation<']
                    example_encoded_labels_npvector = self.__encode_example_label(example) #one-hot-encoding (binary/multi-class) or multi-hot-encoding (multi-label)
                else:
                    example_encoded_labels_npvector = None

                # Could this candidate_pair be fitted into max_seq_len:
                # IF True  --> gather labels stats + convert into machine learning example
                # IF False --> gather labels stats + continue (don't convert to machine learning example)
                if this_example_tokens_indices is None: # means unfitted

                    if generate_output:
                        # gether stats as unfitted
                        if len(example_labels_list) == 0:
                            all_labels_statistics_for_report_dict['neg']['unfitted'] += 1
                        else:
                            # add to the list of unfitted npvectors if it is a POSITIVE example (to be regarded as False-Negative by the system)
                            unfitted_labels_list_of_npvectors.append(example_encoded_labels_npvector)

                            for _lbl_name in example_labels_list:
                                all_labels_statistics_for_report_dict[_lbl_name]['unfitted'] += 1

                    # <<<CRITICAL>>> SKIP THIS PAIR IF CANNOT GENERATE AN EXAMPLE FOR !!!
                    continue

                if generate_output:
                    # add to the list of fitted npvectors
                    fitted_labels_list_of_npvectors.append(example_encoded_labels_npvector)

                    # gether stats as fitted
                    if len(example_labels_list) == 0:
                        all_labels_statistics_for_report_dict['neg']['fitted'] += 1
                    else:
                        for _lbl_name in example_labels_list:
                            all_labels_statistics_for_report_dict[_lbl_name]['fitted'] += 1

                pair = (document_id, example_id , example_e1_id, example_e2_id, this_example_tokens_indices , this_example_tokens_offsets)
                fitted_pairs_tracking.append(pair)  # list of tuples
                fitted_examples_tokens_indices.append(this_example_tokens_indices)  # list of list ...

        # how many pairs did fit into max_seq_len
        number_of_fitted_examples = len(fitted_pairs_tracking)

        # what to return ...
        if number_of_fitted_examples < 1: #<<<CRITICAL>>>
            return {
                "fitted_pairs_tracking"  : None,
                "ANN_inputs"             : None,
                "ANN_outputs"            : None,
                "false_negative_pairs_labels_list_of_npvectors" : unfitted_labels_list_of_npvectors,
                "all_labels_statistics_for_report_dict" : all_labels_statistics_for_report_dict,
            }

        #else: there are some fitted pairs ...
        transformer_input_indices_matrix = np.ones((number_of_fitted_examples, self.BERT_helper.MAX_POSSIBLE_SEQUENCE_LENGTH), dtype=np.int32) * self.model_markers_mapping_to_ids['[PAD]'][0]
        for i in range(number_of_fitted_examples):
            transformer_input_indices_matrix[i, 0:len(fitted_examples_tokens_indices[i])] = fitted_examples_tokens_indices[i]
        ANN_inputs.append(transformer_input_indices_matrix)
        ANN_inputs.append(np.zeros_like(transformer_input_indices_matrix, dtype=np.int8))  # second always all-zero bert input
        if generate_output:
            ANN_outputs.append(np.vstack(fitted_labels_list_of_npvectors))

        return {
            "fitted_pairs_tracking" : fitted_pairs_tracking,
            "ANN_inputs"            : ANN_inputs,
            "ANN_outputs"           : ANN_outputs,
            "false_negative_pairs_labels_list_of_npvectors": unfitted_labels_list_of_npvectors,
            "all_labels_statistics_for_report_dict": all_labels_statistics_for_report_dict,
        }
