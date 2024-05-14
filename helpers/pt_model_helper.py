# PYTORCH HELPER
import os
import sys
import copy

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
import numpy as np

# Code to add the module paths to python
current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))


# -------------------------------------------------------------------------
class BioDeepRelDataset(torch.utils.data.Dataset):
    """
    converts output of helpers.ann_io_generator_cross_sentence_MD.ANN_IO_Generator.generate_ANN_Input_Outputs_pairs(json_data) to something huggingface Trainer/torch can use.
    basically :
    res = ann_io.generate_ANN_Input_Outputs_pairs(json_data)
    ann_inputs = res[1]  (bert inputs)
    ann_outputs = res[2] (labels)
    trainingset = BioDeepRelDataset (ann_inputs, ann_outputs, configs, program_halt, tokenizer)
    """

    def __init__(self, ann_inputs, ann_outputs, configs, program_halt, tokenizer):
        # TODO: implement multi-class and binary classification here
        self.__classification_type = configs['classification_type']
        self.__number_of_classes = configs['RelationTypeEncoding'].number_of_classes
        self.__max_seq_len = ann_inputs[0].shape[1]
        self.__number_of_examples = ann_inputs[0].shape[0]
        self.ann_inputs = ann_inputs
        self.ann_outputs = ann_outputs
        self.tokenizer_outputs = {} # this will hold what type of values the tokenizer returns, e.g. For example: 'input_ids', 'token_type_ids' , 'attention_mask'
        self.pad_token_id = tokenizer.pad_token_id

        # check what fields the tokenizer returns, so we generate the same inputs for the corresponding neural network model.
        # This is needed because some models like DistilBert do not return token_type_ids at all !
        tokenized_test_output = tokenizer("is", add_special_tokens = False, padding='max_length', max_length=10) # e.g: {'input_ids': [284, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

        if not 'input_ids' in tokenized_test_output.keys():
            program_halt("input_ids field was not found in the outputs of the selected tokenizer...")

        for key in tokenized_test_output:
            if key == 'input_ids':
                continue
            if not key in self.tokenizer_outputs:
                self.tokenizer_outputs[key]={}
            self.tokenizer_outputs[key]['subword'] = tokenized_test_output[key][0] #e.g., what is attention_mask value for the real tokens
            self.tokenizer_outputs[key]['pad'] = tokenized_test_output[key][-1] #e.g., what is the attention_mask value for padded tokens

        if self.__classification_type in ("binary", "multi-class") and self.ann_outputs is not None:
            self.ann_out_argmax_label_indices = np.argmax(self.ann_outputs[0], axis=1)

    def __getitem__(self, idx):
        input_ids = self.ann_inputs[0][idx]
        item = {
            'input_ids': torch.tensor(input_ids.tolist()),
        }

        for key in self.tokenizer_outputs.keys():  # for example: 'token_type_ids' , 'attention_mask'
            this_input_key_np_array = np.ones_like(input_ids)
            subwords_indices = np.where(input_ids != self.pad_token_id) #where are real tokens
            pads_indices = np.where(input_ids == self.pad_token_id) #wehre are pad tokens at the end
            this_input_key_np_array[subwords_indices] = self.tokenizer_outputs[key]['subword']
            this_input_key_np_array[pads_indices] = self.tokenizer_outputs[key]['pad']
            item[key]= torch.tensor(this_input_key_np_array.tolist())

        if self.ann_outputs is not None:
            if self.__classification_type == 'multi-label':
                # get all labels being 0 or 1, like [0.,0.,1.,0.,1.,1.,0.,0.,0.,1.,0.] for idx = 5
                # also, float type is needed for BinaryCrossEntropy loss to work
                item['labels'] = torch.tensor(self.ann_outputs[0][idx], dtype=torch.float)
            else:  # binary, multi-class #get argmax, like [6]
                # item['labels'] = torch.tensor(np.argmax(self.ann_outputs[0][idx], axis=1)) #this did not work for idx=6, but works for example for idx = 4:6 or :6
                item['labels'] = torch.tensor(self.ann_out_argmax_label_indices[idx])

        return item

    def __len__(self):
        return self.__number_of_examples


class BioDeepRelCustomCallback(TrainerCallback):
    def __init__(self, trainer, relation_extraction_pipeline_pt_object) -> None:
        super().__init__()
        self._trainer = trainer
        self.lp = relation_extraction_pipeline_pt_object.lp
        self.relation_extraction_pipeline_pt_object = relation_extraction_pipeline_pt_object

    def on_epoch_begin(self, args, state, control, **kwargs):
        msg = ["*" * 80]
        try:
            epoch_counter = str(int(state.epoch))
            msg += ["Training - On_Epoch_Begin ... Epoch :" + epoch_counter]
        except:
            msg += ["Training - On_Epoch_Begin ..."]
        self.lp(msg)

    def on_epoch_end(self, args, state, control, **kwargs):
        control_copy = copy.deepcopy(control)
        if len(state.log_history) > 0:
            self.lp(["Training - On_Epoch_Log : " + str(state.log_history[-1]), "*" * 80])

        if self.relation_extraction_pipeline_pt_object is not None:
            self.relation_extraction_pipeline_pt_object.PREDICT_EVALUATE_WRITEBACK_CALLBACK()

        """
        #copy from internet 
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
        """
        return control_copy  # Totally unsure if this is needed. Just copy/pasting from internet examples: if you do evaluate in some sort, return control_copy ...

    def on_log(self, args, state, control, **kwargs):
        pass


class MultilabelTrainer(Trainer):
    # Trainer with special loss function for multi-label classification for huggingface Trainer
    # Following https://huggingface.co/transformers/main_classes/trainer.html
    # Following https://discuss.huggingface.co/t/fine-tune-for-multiclass-or-multilabel-multiclass/4035/8
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


class PT_helper:
    def __init__(self, lp, program_halt, pipeline_configs, arg_model_name, arg_use_fast_tokenizer=True,
                 max_seq_len=512):
        # TODO: implement multi-class and binary classification here
        # Remove once everything is finished

        self.lp = lp
        self.program_halt = program_halt
        self.model_name = arg_model_name
        self.pipeline_configs = pipeline_configs
        self.MAX_POSSIBLE_SEQUENCE_LENGTH = max_seq_len
        self.trainer = None

        self.__classification_type = self.pipeline_configs['classification_type']
        self.__number_of_classes = self.pipeline_configs['RelationTypeEncoding'].number_of_classes

        if self.__classification_type in ('multi-class', 'binary'):
            self.DefaultTrainerClass = Trainer
        else:
            self.DefaultTrainerClass = MultilabelTrainer

        if not (10 <= max_seq_len <= 512):
            self.program_halt("max_seq_len should be >= 10 and <= 512. ")

        if not isinstance(arg_model_name, str):
            self.program_halt("invalid model_name. model_name should be string.")

        try:
            self.lp("BACKEND:\t1-Initializing AutoConfig : " + arg_model_name + " ...")
            if self.__classification_type == 'multi-label':
                self.config = AutoConfig.from_pretrained(arg_model_name,
                                                         output_hidden_states=False,
                                                         num_labels=self.__number_of_classes,
                                                         label2id=pipeline_configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping,
                                                         id2label=pipeline_configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping,
                                                         problem_type="multi_label_classification")
            else:  # binary, multi-class
                self.config = AutoConfig.from_pretrained(arg_model_name,
                                                         output_hidden_states=False,
                                                         num_labels=self.__number_of_classes,
                                                         label2id=pipeline_configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping,
                                                         id2label=pipeline_configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping)

            self.config.return_dict = True
            self.lp("\tsuccessful.")
            if self.MAX_POSSIBLE_SEQUENCE_LENGTH > self.config.max_position_embeddings:
                msg = ["max_seq_len is greater than config.max_position_embeddings, which is " + str(self.config.max_position_embeddings)]
                msg += ["setting MAX_POSSIBLE_SEQUENCE_LENGTH to : " + str(self.config.max_position_embeddings)]
                self.lp(msg)
                self.MAX_POSSIBLE_SEQUENCE_LENGTH = self.config.max_position_embeddings

            # self.tokenizer = self.tokenizer
        except Exception as E:
            self.lp(sys.path)
            self.program_halt("Error loading the model. Error: " + str(E))

        try:
            self.lp("BACKEND:\t2-Initializing AutoTokenizer from : " + arg_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(arg_model_name, config=self.config, use_fast=arg_use_fast_tokenizer)
            self.lp("\tsuccessful.")

            # <<<CRITICAL>>> add additional unused tokens as special tokens...
            _special_tokens_added = False
            if not "[unused1]" in self.tokenizer.vocab:
                _special_tokens_added = True
                self.lp("BACKEND:\t3-Adding special tokens ... ")
                additional_unused_tokens = ["[unused" + str(i) + "]" for i in range(1, 51)]
                self.tokenizer.add_special_tokens({"additional_special_tokens": additional_unused_tokens})
                # _total_tokens = len(self.tokenizer.vocab)
                # self.config.vocab_size = _total_tokens #this will fail, because of inconsistency between the model defined in ram when model is being loaded from disk.
                self.lp("\tsuccessful.")
            else:
                self.lp("BACKEND:\t3-Special tokens, e.g. [unused1] successfully found in the tokenizer ...")

        except Exception as E:
            self.program_halt("Error in AutoTokenizer. Error: " + str(E))

        try:
            self.lp("BACKEND:\t4-Loading pretrained model ... ")
            self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(arg_model_name, config=self.config)
            # self.pretrained_model = AutoModelForSequenceClassification.from_config(self.config) # maybe this is not a good idea: see https://github.com/huggingface/transformers/issues/4685

            if _special_tokens_added:
                _number_of_tokens = len(self.tokenizer.vocab)
                # <<<CRITICAL>>> see: https://stackoverflow.com/questions/65683013/indexerror-index-out-of-range-in-self-while-try-to-fine-tune-roberta-model-afte
                # value of word_embeddings.padding_idx is lost after resize! therefore, we need to set it again with its previous value!
                VALUE_OF_padding_idx = self.pretrained_model.base_model.embeddings.word_embeddings.padding_idx
                self.pretrained_model.resize_token_embeddings(_number_of_tokens)
                self.pretrained_model.base_model.embeddings.word_embeddings.padding_idx = VALUE_OF_padding_idx # <<<CRITICAL>>>
            self.lp("\tsuccessful.")
        except Exception as E:
            self.lp("Error in loading model. Error: " + str(E))
            self.lp("Trying to load with from_tf=True")
            try:
                self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(arg_model_name, config=self.config, from_tf=True)
                self.lp("\tsuccessful.")
            except Exception as E:
                self.program_halt("Error in loading model. Error: " + str(E))

    def tokenize_text(self, text, add_start_end_tokens=False):
        if add_start_end_tokens:
            return [self.tokenizer.cls_token] + self.tokenizer.tokenize(text) + [self.tokenizer.sep_token] #use correct CLS and SEP tokens based on this particular model vocab!
        else:
            return self.tokenizer.tokenize(text)

    def vectorize_without_taking_max_len_into_account(self, sentence_tokens_list):
        """
        This is for CROSS_SENTENCE example and ann_io generation, for the whole document.
        NOTES:
            1) It DOES NOT take into account the bert max_len.
            2) It DOES NOT add [CLS] or [SEP]
        """
        return self.tokenizer.convert_tokens_to_ids(sentence_tokens_list)

    def __evaluate_training_arguments(self):
        # 1-optimizer type
        if not "optimizer" in self.training_arguments:
            self.program_halt("optimizer not found in training_arguments.")
        if not self.training_arguments["optimizer"] in ["adam", "nadam", "adamwarmup", "sgd"]:
            self.program_halt("optimizer in training_arguments should be either of adam , nadam, adamwarmup or sgd")

        if self.training_arguments['optimizer'] != "adam":
            self.lp("[WARNING]: for now, only adam optimizer and its variants are supported. Switching to adam.")

        del self.training_arguments["optimizer"]

        # <<<CRITICAL>>> NEW ...
        """
        without setting optimizer, default used to be: optim=OptimizerNames.ADAMW_HF in the TrainingArguments.
        now I set it to torch.optim.AdamW 

        not to get error: 
        /opt/rh/rh-python38/root/usr/local/lib/python3.8/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
          warnings.warn(
          File "/opt/rh/rh-python38/root/usr/local/lib/python3.8/site-packages/transformers/trainer.py", line 1264, in train
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
          File "/opt/rh/rh-python38/root/usr/local/lib/python3.8/site-packages/transformers/trainer.py", line 829, in create_optimizer_and_scheduler
            self.create_optimizer()
          File "/opt/rh/rh-python38/root/usr/local/lib/python3.8/site-packages/transformers/trainer.py", line 862, in create_optimizer
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
          File "/opt/rh/rh-python38/root/usr/local/lib/python3.8/site-packages/transformers/optimization.py", line 318, in __init__
            if not 0.0 <= eps:
        TypeError: '<=' not supported between instances of 'float' and 'tuple'

        See: https://discuss.huggingface.co/t/huggingface-transformers-longformer-optimizer-warning-adamw/14711

        VERY CRITICAL: https://github.com/huggingface/transformers/issues/14539
        """
        self.training_arguments["optim"] = "adamw_torch"  # <<<CRITICAL>>> NEW ...
        # THIS ONLY WORKS for tensorflow > 1.9 on puhti

        # 2-learning_rate
        if not "learning_rate" in self.training_arguments:
            self.program_halt("learning_rate not found in training_arguments.")
        if not isinstance(self.training_arguments["learning_rate"], float):
            self.program_halt("learning_rate in training_arguments should be float and > 0")
        if self.training_arguments["learning_rate"] <= 0:
            self.program_halt("learning_rate in training_arguments should be float and > 0")

        # 3-minibatch_size
        if not "minibatch_size" in self.training_arguments:
            self.program_halt("minibatch_size not found in training_arguments.")
        try:
            self.training_arguments["minibatch_size"] = int(self.training_arguments["minibatch_size"])
            if self.training_arguments["minibatch_size"] <= 1:
                raise Exception()
            self.training_arguments["per_device_train_batch_size"] = self.training_arguments["minibatch_size"]
            self.training_arguments["per_device_eval_batch_size"] = self.training_arguments["minibatch_size"]
            del self.training_arguments['minibatch_size']
        except:
            self.program_halt("minibatch_size in training_arguments should be int and >= 1")

        # 4-num_train_epochs
        try:
            self.training_arguments["num_train_epochs"] = float(self.training_arguments["num_train_epochs"])
            if self.training_arguments["num_train_epochs"] < 1:
                raise Exception()
        except:
            self.program_halt("num_train_epochs in training_arguments should be int and >= 1")

        # set important properties
        self.training_arguments['save_strategy'] = 'no'
        self.training_arguments['evaluation_strategy'] = 'no'
        self.training_arguments['logging_strategy'] = "epoch"
        self.training_arguments['disable_tqdm'] = True

        self.training_arguments = TrainingArguments(**self.training_arguments)
        self.lp(str(self.training_arguments))

    def build_model(self, training_arguments):
        self.training_arguments = training_arguments
        self.__evaluate_training_arguments()

    def train(self, trainingset_ann_inputs, trainingset_ann_outputs, relation_extraction_pipeline_pt_object):
        if self.trainer is not None:
            self.program_halt("self.traininer is not None. You cannot call train function again.")

        trainingset_dataset = BioDeepRelDataset(trainingset_ann_inputs,
                                                trainingset_ann_outputs,
                                                self.pipeline_configs,
                                                self.program_halt,
                                                self.tokenizer)

        self.trainer = self.DefaultTrainerClass(self.pretrained_model,
                                                self.training_arguments,
                                                train_dataset=trainingset_dataset,
                                                eval_dataset=None)

        self.trainer.add_callback(BioDeepRelCustomCallback(self.trainer, relation_extraction_pipeline_pt_object))
        self.trainer.train()

    def build_load_trainer_for_prediction(self):
        if self.trainer is not None:
            self.program_halt("self.traininer is not None. You cannot call build_load_trainer_for_prediction function.")
        self.trainer = self.DefaultTrainerClass(self.pretrained_model)

    def predict(self, ann_inputs):
        # Rules:
        # In multi-class and binary: argmax(softmax(logits)) == argmax(logits)
        # In multi-label: sigmoid(logits) > 0.5 <=> logits > 0
        # However, since the original evaluation and writeback are written based on the outputs being transfered through sigmoid or softmax,
        # I convert the logits to either based on the type of the task. this makes system combination easier.

        if self.trainer is None:
            self.program_halt("self.traininer is None! You should first train the model, or load a model.")

        test_set = BioDeepRelDataset(ann_inputs, None, self.pipeline_configs, self.program_halt, self.tokenizer)
        results = self.trainer.predict(test_set)
        prediction_logits = results.predictions

        if self.__classification_type == 'multi-label':
            prediction_output = torch.sigmoid(torch.tensor(prediction_logits)).numpy()
        else:  # binary, multi-class
            prediction_output = torch.softmax(torch.tensor(prediction_logits), axis=1).numpy()
        return prediction_output


    def predict_return_logits(self, ann_inputs):
        # Rules:
        # In multi-class and binary: argmax(softmax(logits)) == argmax(logits)
        # In multi-label: sigmoid(logits) > 0.5 <=> logits > 0
        # However, since the original evaluation and writeback are written based on the outputs being transfered through sigmoid or softmax,
        # I convert the logits to either based on the type of the task. this makes system combination easier.

        if self.trainer is None:
            self.program_halt("self.traininer is None! You should first train the model, or load a model.")
        test_set = BioDeepRelDataset(ann_inputs, None, self.pipeline_configs, self.program_halt, self.tokenizer)
        results = self.trainer.predict(test_set)
        prediction_logits = results.predictions
        return prediction_logits
