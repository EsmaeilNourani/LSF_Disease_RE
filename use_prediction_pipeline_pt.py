import os
import sys
import json
import shutil
import datetime
import gzip
import tarfile
import torch
import numpy as np

from helpers import logger
from helpers import configs_manager
from helpers import general_helpers as ge
from helpers import pt_model_helper
from helpers import brat_json_converter
from helpers import example_generation_cross_sentence_MD
from helpers import ann_io_generator_cross_sentence_MD as ann_io_generator
from helpers import pipeline_variables

class LargeScaleTorchRelModelHelper:
    def __init__(self,
                 lp,
                 configs,
                 program_halt,
                 program_halt_raise_exception_do_not_exit,
                 pretrained_re_model_folder_path,
                ):

        #1. init variables
        self.lp = lp
        self.configs = configs
        self.program_halt = program_halt
        self.program_halt_raise_exception_do_not_exit = program_halt_raise_exception_do_not_exit
        self.pretrained_re_model_folder_path = pretrained_re_model_folder_path

        #2. create other variables, load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lp ("torch device: " + str(self.device))
        self.__load_re_model_from_folder()

    def __load_re_model_from_folder(self):
        if self.pretrained_re_model_folder_path[-1] != "/":
            self.pretrained_re_model_folder_path += "/"

        self.lp("-"*40 + " LOADING MODEL " + "-"*40)
        self.lp("\t1. Checking pretrained_re_model_folder_path: " + str(self.pretrained_re_model_folder_path))
        if not os.path.isdir(self.pretrained_re_model_folder_path):
            self.program_halt("invalid path given for pretrained_re_model_folder_path :" + str(self.pretrained_re_model_folder_path))

        self.lp("\t2. loading info.json from the folder.")
        try:
            info_json_file_path = self.pretrained_re_model_folder_path + "info.json"
            with open(info_json_file_path, "rt", encoding='utf-8') as f:
                self.__info_json_file_content = json.load(f)
            self.lp(["\tinfo.json: " + os.path.abspath(info_json_file_path),
                     "\tfile contents:", "-" * 20,
                     json.dumps(self.__info_json_file_content, indent=4, sort_keys=True), "-" * 80])
            self.max_seq_len = int(self.__info_json_file_content['max_seq_len'])
            self.representation_strategy = eval("pipeline_variables." + self.__info_json_file_content['representation_strategy'])
        except Exception as E:
            self.program_halt("Error: " + str(E))

        self.lp("\t3. loading and building model.")
        #<<<CRITICAL>>> passing program_halt_raise_exception_do_not_exit instead of program_halt, so that the pipeline just raises an exception and not totally crash
        self.torch_helper = pt_model_helper.PT_helper(self.lp, self.program_halt_raise_exception_do_not_exit, self.configs, self.pretrained_re_model_folder_path, max_seq_len=self.max_seq_len)
        self.torch_helper.build_load_trainer_for_prediction()
        self.brat_json_converter = brat_json_converter.brat_json_Converter(self.lp, self.program_halt_raise_exception_do_not_exit, self.configs)
        self.example_generator = example_generation_cross_sentence_MD.example_generator(self.lp, self.program_halt_raise_exception_do_not_exit, self.configs)
        self.ann_input_output_generator = ann_io_generator.ANN_IO_Generator(self.lp, self.program_halt_raise_exception_do_not_exit, self.configs, self.torch_helper)

        self.torch_helper.pretrained_model.eval() #<<<CRITICAL>>> THIS IS NEEDED TO TURN OFF DROPOUT and OTHER REGULARIZATIONS
        #self.torch_helper.pretrained_model.zero_grad() #<<<CRITICAL>>> DO NOT DO THIS.. NOT In training ...

    def encode_brat_document(self, txt_file_path, ann_file_path, dont_generate_negatives_if_sentence_distance_ge):
        try:
            json_data = self.brat_json_converter.convert_brat_to_json(txt_file_path, ann_file_path, output_file_json=None, all_event_types=[], encoding="utf-8")
        except Exception as E:
            the_ann_file_name = ann_file_path.split("/")[-1]
            errmsg = "error in encode_brat_document/brat_json_converter. ann: " + the_ann_file_name + " .[INTERNAL_DETAIL] :" + str(E)
            raise Exception(errmsg)

        try:
            json_data, counter_annotated_positives, counter_generated_positives, counter_generated_negatives = self.example_generator.generate_examples(
                input_json_fileaddress=None,
                output_json_fileaddress=None,
                input_json_conent=json_data,
                dont_generate_negatives_if_sentence_distance_ge=dont_generate_negatives_if_sentence_distance_ge) #<<<CRITICAL>>> set this to None to always try to generate anything. in our case, it is not important cause seq_len=128 and we're just processing those examples which are predicted to be positive ...
        except Exception as E:
            the_ann_file_name = ann_file_path.split("/")[-1]
            errmsg = "error in encode_brat_document/example_generator. ann: " + the_ann_file_name + " .[INTERNAL_DETAIL] :" + str(E)
            raise Exception(errmsg)

        if counter_generated_positives + counter_generated_negatives < 1:
            return json_data, None, None, None

        try:
            pair_tracking, ann_inputs, ann_outputs = self.ann_input_output_generator.generate_ANN_Input_Outputs_pairs(
                json_data,
                generate_output=True,
                strategy=self.representation_strategy)
        except Exception as E:
            the_ann_file_name = ann_file_path.split("/")[-1]
            errmsg = "error in encode_brat_document/ann_input_output_generator. ann: " + the_ann_file_name + " .[INTERNAL_DETAIL] :" + str(E)
            raise Exception(errmsg)

        if pair_tracking is None:  # because of max_seq_len, maybe some examples are discarded. Then, a situation may arise that pair_tracking is None.
            return json_data, None, None, None

        return json_data, pair_tracking, ann_inputs, ann_outputs


class LargeScalePredictionPipeline_torch(object):
    def __init__(self,
                 configs_file_path,
                 pretrained_model_path,
                 log_file_path,
                 input_folder_path,
                 output_folder_path,
                 create_output_ann_files=False,
                 dont_generate_negatives_if_sentence_distance_ge=None,
                 write_logits_to_output_tsv=True):

        #1. init local variables
        self.configs_file_path = configs_file_path
        self.pretrained_model_path = pretrained_model_path
        self.log_file_path = log_file_path
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.create_output_ann_files = create_output_ann_files
        self.dont_generate_negatives_if_sentence_distance_ge = dont_generate_negatives_if_sentence_distance_ge
        self.write_logits_to_output_tsv = write_logits_to_output_tsv

        #2. check the path for log and configs files
        if os.path.isfile(log_file_path):
            print ("log file already exists. deleting and recreating the file :" + log_file_path)
            os.remove(log_file_path)
        if not os.path.isfile(configs_file_path):
            print("[ERROR] configs file not found: " + configs_file_path + "\nHALTING.")
            sys.exit(0)

        #3. create lp and configs objects
        self.logger = logger.Logger(log_file_path)
        self.lp = self.logger.lp
        self.configs = configs_manager.ConfigsManager(configs_file_path, self.lp, self.program_halt).configs

        #4. check input/output folders, add '/' to the end if needed
        self.__check_input_output_folders()

        #5. set temp folder path here
        self.temp_folder_path = self.output_folder_path + "tmp" + "_" + datetime.datetime.now().strftime("%Y%M%d_%H%M%S") + "/"
        self.__recreate_temp_folder()

        #6. log parameters
        msg_params = ["Running pipeline:", "-" * 80,
                      "\t- configs_file_path                               : " + str(self.configs_file_path),
                      "\t- pretrained_model_name_or_path                   : " + str(self.pretrained_model_path),
                      "\t- log_file_path                                   : " + str(self.log_file_path),
                      "\t- input_folder_path                               : " + self.input_folder_path,
                      "\t- output_folder_path                              : " + self.output_folder_path,
                      "\t- create_output_ann_files                         : " + str(self.create_output_ann_files),
                      "\t- temp_folder_path                                : " + self.temp_folder_path,
                      "\t- dont_generate_negatives_if_sentence_distance_ge : " + str(self.dont_generate_negatives_if_sentence_distance_ge),
                      "-" * 80]
        self.lp(msg_params)

        #7. load pretrained relation extraction model ...
        self.ls_pt_rel_model_helper = LargeScaleTorchRelModelHelper(
            self.lp,
            self.configs,
            self.program_halt,
            self.program_halt_raise_exception_do_not_exit,
            self.pretrained_model_path)

    def program_halt(self, message):
        #THIS WILL HALT and CRASH (runs self.exit function) THE WHOLE PIPELINE
        self.logger.lp_halt(message)
        self.exit()

    def program_halt_raise_exception_do_not_exit(self, message):
        #THIS WILL NOT HALT, but just raises exception, to be caught, and then we will proceed to the next input file ...
        #THIS WILL BE PASSED FOR EXAMPLE to brat_json_converter etc so that bad .ann+.txt files don't crash the whole pipeline, and we skip to the next file
        raise Exception(message)

    def exit(self):
        if self.logger.is_open():
            self.lp ("EXITING PROGRAM ... ")
            self.logger.close()
        sys.exit(0)

    def __check_input_output_folders(self):
        if not isinstance(self.input_folder_path, str):
            self.program_halt("invalid path for input_folder_path :" + str(self.input_folder_path))
        if not isinstance(self.output_folder_path, str):
            self.program_halt("invalid path for output_folder_path :" + str(self.output_folder_path))
        if self.input_folder_path[-1] != "/":
            self.input_folder_path += "/"
        if self.output_folder_path[-1] != "/":
            self.output_folder_path += "/"
        if not os.path.isdir(self.input_folder_path):
            self.program_halt("invalid path for input_folder_path :" + str(self.input_folder_path))
        if not os.path.isdir(self.output_folder_path):
            self.program_halt("invalid path for output_folder_path :" + str(self.output_folder_path))

    def __recreate_temp_folder(self):
        try:
            if shutil.os.path.exists(self.temp_folder_path):
                shutil.rmtree(self.temp_folder_path)
            shutil.os.makedirs(self.temp_folder_path)
        except Exception as E:
            self.program_halt('unable to create temp folder here :' + self.temp_folder_path + "\nError: " + str(E))

    def __mkdir(self, folder_address):
        if not shutil.os.path.exists(folder_address):
            try:
                shutil.os.makedirs(folder_address)
            except Exception as E:
                self.program_halt("could not create folder : " + folder_address + "\nerror: " + str(E))

    def __rm_directory_with_content(self,folder_address):
        if shutil.os.path.exists(folder_address):
            try:
                shutil.rmtree(folder_address)
            except Exception as E:
                self.program_halt("could not remove folder : " + folder_address + "\nerror: " + str(E))

    def run_large_scale_pipeline(self):
        #PROCESS ALL tar.gz files in a given input folder ...
        for input_tar_gz_file_path in sorted(ge.get_all_files_with_extension(self.input_folder_path , "tar.gz")):
            self.__recreate_temp_folder()
            date_time_start = datetime.datetime.now()
            self.lp ("input tar.gz: " + input_tar_gz_file_path)
            input_tar_gz_file_name = input_tar_gz_file_path.split("/")[-1]
            output_gzip_file_path = self.output_folder_path + input_tar_gz_file_name.split(".tar.gz")[0] + ".outtsv.gz"
            output_errlog_file_path = self.output_folder_path + input_tar_gz_file_name.split(".tar.gz")[0] + ".err.log"

            # 1: create output ann folder if requested
            try:
                if self.create_output_ann_files:
                    output_ann_folder = self.output_folder_path + input_tar_gz_file_name.split(".tar.gz")[0] + "/"
                    self.__rm_directory_with_content(output_ann_folder)
                    self.__mkdir(output_ann_folder)
                else:
                    output_ann_folder = None
            except Exception as E:
                errmsg = "[ERROR#1]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_DETAILS]: could not create output ann folder :" + output_ann_folder + " .\t" + \
                         "[MORE_DETAILS]: " + str(E)
                self.lp (errmsg)
                date_time_end = datetime.datetime.now()
                self.lp ("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

            # 2: create error_log handler
            try:
                if os.path.isfile(output_errlog_file_path):
                    os.remove(output_errlog_file_path)
                this_input_logger = logger.Logger(output_errlog_file_path, write_header_to_file=False)
                errlp = this_input_logger.lp
            except Exception as E:
                errmsg = "[ERROR#2]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_DETAILS]: could not create log file :" + output_errlog_file_path + " .\t" + \
                         "[MORE_DETAILS]: " + str(E)
                self.lp (errmsg)
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

            # 3: create output_gz file handler
            try:
                #3.1 create file handle
                output_gz_file_handler = None
                output_gz_file_handler = gzip.open(output_gzip_file_path, "wt", encoding='utf-8')

                #3.2 writing header into tsv.gz file: pmid, e1, e2, and then logits for relation types...
                all_relation_type_indices_sorted = sorted(self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping.keys())
                all_relation_typr_names = [self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping[i] for i in all_relation_type_indices_sorted]
                output_gz_msg_lst = ["pmid" , "e1" , "e2"] + all_relation_typr_names
                output_gz_msg_str = "#" + "\t".join(output_gz_msg_lst) + "\n"
                output_gz_file_handler.write(output_gz_msg_str)
                output_gz_file_handler.flush()

            except Exception as E:
                errmsg = "[ERROR#3]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_DETAILS]: could not create output file :" + output_gzip_file_path + " .\t" + \
                         "[MORE_DETAILS]: " + str(E)
                self.lp(errmsg)
                errlp(errmsg)
                this_input_logger.close()
                if output_gz_file_handler is not None:
                    try:
                        output_gz_file_handler.close()
                    except:
                        pass
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

            # 4: process input tar.gz, write into output .gz and possibly into .ann files
            try:
                self.__process_one_input_tar_gz(input_tar_gz_file_path, output_gz_file_handler, output_ann_folder, errlp)
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                errlp("<<<END_OF_PROCESSING>>>")
                try:
                    output_gz_file_handler.close()
                    this_input_logger.close()
                except:
                    pass
            except Exception as E:
                #dont use finally here to wait it out until the loop is done ...
                errmsg = "[ERROR#4]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_DETAILS]: processing input tar.gz file.\t" + \
                         "[MORE_DETAILS]: " + str(E)
                self.lp(errmsg)
                errlp(errmsg)
                try: #don't wait this out with finally till the loop ends, do it now here!
                    output_gz_file_handler.close()
                    this_input_logger.close()
                except:
                    pass
                date_time_end = datetime.datetime.now()
                self.lp("[time-delta]\t" + input_tar_gz_file_name + "\t " + str(date_time_end - date_time_start))
                continue

        #finalize work ...
        self.lp("<<<END_OF_PROCESSING_ALL_BATCHES>>>")
        self.__rm_directory_with_content(self.temp_folder_path) #delete tmp folder ...

    def __process_one_input_tar_gz(self, input_tar_gz_file_path, output_gz_file_handler, output_ann_folder, errlp):
        #1. check integrity of the input .tar.gz file: check .ann+.txt
        try:
            anns = []
            txts = []
            tar = None
            tar = tarfile.open(input_tar_gz_file_path , "r:gz", encoding='utf-8')
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                file_name = member.name.split("/")[-1]
                if file_name.endswith(".ann"):
                    anns.append(file_name.split(".ann")[0])
                elif file_name.endswith(".txt"):
                    txts.append(file_name.split(".txt")[0])

            anns = set(anns)
            txts = set(txts)
            if len(anns) != len(txts):
                anns_minues_txts = anns - txts
                anns_minues_txts_len = len(anns_minues_txts)
                if anns_minues_txts_len > 0:
                    raise Exception("There are " + str(anns_minues_txts_len) + " .ann files with no corresponding .txt :" + str(sorted(anns_minues_txts)))

                txts_minues_anns = txts - anns
                txts_minues_anns_len = len(txts_minues_anns)
                if txts_minues_anns_len > 0:
                    raise Exception("There are " + str(txts_minues_anns_len) + " .txt files with no corresponding .ann :" + str(sorted(txts_minues_anns)))

            if len(anns) < 1:
                raise Exception("There are zero ann files in the input file.")

        except Exception as E:
            if tar is not None:
                if not tar.closed:
                    tar.close()
            raise Exception(str(E))

        #2. Unpack each brat doc (.ann+.txt pair) from THIS INPUT tar.gz file and process them one-by-one ...
        for brat_file in sorted(anns):
            try: #if an exception happens in this try, we don't raise an exception for the WHOLE .tat.gz, but we simply skip this brat.
                # We log into corresponding log file of the .tar.gz and continue with the next brat document ...

                # 2.1: extract .ann+.txt from .tar.gz into temp
                ann_file_path = self.temp_folder_path + brat_file + ".ann" #to hold file-name for later use after extraction from .tar
                txt_file_path = self.temp_folder_path + brat_file + ".txt" #to hold file-name for later use after extraction from .tar
                tar.extract("./" + brat_file + ".ann", self.temp_folder_path)
                tar.extract("./" + brat_file + ".txt", self.temp_folder_path)
                pm_id = brat_file

                # 2.2: create empty output .ann file if requested
                if output_ann_folder is not None:
                    output_ann_file_handler = open(output_ann_folder + brat_file + ".ann", "wt", encoding='utf-8')
                else:
                    output_ann_file_handler = None

                # 2.3 encode brat document , through an exception if anything goes wrong , and we will continue with the next brat document (.ann/.txt)
                json_data, pair_tracking, ann_inputs, ann_outputs = self.ls_pt_rel_model_helper.encode_brat_document(
                    txt_file_path,
                    ann_file_path,
                    dont_generate_negatives_if_sentence_distance_ge=self.dont_generate_negatives_if_sentence_distance_ge)

                os.remove(ann_file_path)
                os.remove(txt_file_path)

                # 2.4 write entities into .ann file if .ann file creation is requested ...
                if output_ann_file_handler is not None:
                    self.__write_entities_to_ann(output_ann_file_handler, json_data)

                # 2.5 check if there are any valid examples (candidate entity pairs) found in the file, based on .ann info and MAX_SEQ_LEN
                if pair_tracking is None:
                    if output_ann_file_handler is not None:
                        try:
                            output_ann_file_handler.close()
                        except:
                            pass
                    continue #proceed to the next brat document (.ann+.txt) in the input .tar.gz file

                # 2.6 predict
                y_pred_logits = self.ls_pt_rel_model_helper.torch_helper.predict_return_logits(ann_inputs) # y_pred.shape --> (number of examples, conf_vector_dim) and y_pred[0,:] --> confs for first example.

                # 2.7 write the results into output .tsv.gz
                self.__wirte_output_to_tsv_gz(output_gz_file_handler, pm_id, pair_tracking, y_pred_logits)

                # 2.8 write the results into output .ann if requested and close
                if output_ann_file_handler is not None:
                    self.__write_output_to_ann(output_ann_file_handler, pair_tracking, y_pred_logits)
                    try:
                        output_ann_file_handler.close()
                    except:
                        pass

            except Exception as E:
                #e1: log into local .tar.gz log-file
                errmsg = "[ERROR]: input tar.gz : " + input_tar_gz_file_path + " .\t" + \
                         "[ERROR_ANN] : " + brat_file + ".ann \t" + \
                         "[ERROR_DETAILS] :" + str(E)
                errlp(errmsg)

                #e2: close output ann_file_handler if is open
                if output_ann_file_handler is not None:
                    try:
                        output_ann_file_handler.flush()
                        output_ann_file_handler.close()
                    except:
                        pass

                #e3: remove temp files
                try:
                    if os.path.isfile(ann_file_path):
                        os.remove(ann_file_path)
                    if os.path.isfile(txt_file_path):
                        os.remove(txt_file_path)
                except:
                    pass

                continue

    def __wirte_output_to_tsv_gz(self, output_gz_file_handler, pm_id, pair_tracking, y_pred_logits):
        #we write logits to output .tsv files, unless softmax or sigmoid output is requested ...
        if self.write_logits_to_output_tsv == False:
            if self.configs['classification_type'] in ["binary", "multi-class"]:
                y_pred_logits = torch.softmax(torch.tensor(y_pred_logits), axis=1).numpy()
            else:  # multi-label
                y_pred_logits = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()

        try:
            for pair_info, logit_vector in zip(pair_tracking, y_pred_logits):
                e1_id = pair_info[2]
                e2_id = pair_info[3]
                gz_output_list = [pm_id, e1_id, e2_id] + [str(i) for i in logit_vector.tolist()]
                gz_output_str = "\t".join(gz_output_list) + "\n"
                output_gz_file_handler.write(gz_output_str)
                output_gz_file_handler.flush()
        except Exception as E:
            raise Exception("error in __wirte_output_to_tsv_gz: " + str(E))

    def __write_output_to_ann(self, output_ann_file_handler, pair_tracking, y_pred_logits):
        relation_id_tracker = 1 #for creating R1, R2, ... in the .ann file
        try:
            if self.configs['classification_type'] in ["binary", "multi-class"]:
                y_pred_labels_indices = np.argmax(y_pred_logits, axis=1)
                for index, pair_info in enumerate(pair_tracking):
                    #get info
                    e1_id = pair_info[2]
                    e2_id = pair_info[3]
                    predicted_relation_type = self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping[y_pred_labels_indices[index]]

                    #discard if negative
                    if predicted_relation_type == 'neg':
                        continue

                    #if it is a directed relation type, swap e1, e2 if reverese:
                    if predicted_relation_type[-1] == ">":
                        predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation>' --> 'Regulation'
                    elif predicted_relation_type[-1] == "<":
                        predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation<'  --> 'Regulation'
                        e1_id, e2_id = e2_id, e1_id #SWAP ... swap does not ruin anything, since there is only one label to be written ... for multi-label we use intermediate variables.

                    #get new relation id
                    relation_id = "R" + str(relation_id_tracker)
                    relation_id_tracker += 1

                    #write to file
                    output_string = relation_id + "\t" + predicted_relation_type + " Arg1:" + e1_id + " Arg2:" + e2_id + "\n"
                    output_ann_file_handler.write(output_string)
                    output_ann_file_handler.flush()

            else: # multi-label
                for index, pair_info in enumerate(pair_tracking):
                    e1_id = pair_info[2]
                    e2_id = pair_info[3]
                    positive_label_indices = list(np.nonzero(y_pred_logits[index, :] >= 0.5)[0])  # --> for example something like [0,4,5] or [4] or []

                    for positive_label_index in positive_label_indices:
                        predicted_relation_type = self.configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping[positive_label_index]

                        if predicted_relation_type[-1] == ">":
                            writeback_e1, writeback_e2 = e1_id, e2_id
                            predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation>' --> 'Regulation'
                        elif predicted_relation_type[-1] == "<":
                            writeback_e1, writeback_e2 = e2_id, e1_id  # <<<CRITICAL>>> SWAP e1 and e2
                            predicted_relation_type = predicted_relation_type[:-1]  # 'Regulation<'  --> 'Regulation'
                        else:
                            writeback_e1, writeback_e2 = e1_id, e2_id

                        #get new relation id
                        relation_id = "R" + str(relation_id_tracker)
                        relation_id_tracker += 1

                        # write to file
                        output_string = relation_id + "\t" + predicted_relation_type + " Arg1:" + writeback_e1 + " Arg2:" + writeback_e2 + "\n"
                        output_ann_file_handler.write(output_string)
                        output_ann_file_handler.flush()

        except Exception as E:
            raise Exception("error in __write_output_to_ann: " + str(E))

    def __write_entities_to_ann(self, output_ann_file_handler, json_data):
        try:
            for document in json_data['documents']:
                for entity_id in document['entities'].keys():
                    entity_info = document['entities'][entity_id]  # dictionary
                    output_string = entity_id + "\t" + entity_info['tag'] + " "
                    for span in entity_info['orig_spans']:
                        output_string += str(span[0]) + " " + str(span[1]) + ";"
                    output_string = output_string[:-1]
                    output_string += "\t" + entity_info['text'] + "\n"
                    output_ann_file_handler.write(output_string)
                    output_ann_file_handler.flush()
        except Exception as E:
            raise Exception("error in __write_entities_to_ann: " + str(E))



