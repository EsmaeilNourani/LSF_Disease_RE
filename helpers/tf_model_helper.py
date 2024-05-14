import os
import sys
import math

#Code to add the module paths to python
current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))
#-------------------------------------------------------------------------

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense
import tensorflow.keras.optimizers
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from transformers.optimization_tf import create_optimizer
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

class TF_helper:
    def __init__(self, lp, program_halt, pipeline_configs, arg_model_name, arg_use_fast_tokenizer=True, max_seq_len=512, tokenizer_folder_path=None):
        self.lp = lp
        self.program_halt = program_halt
        self.model_name = arg_model_name
        self.pipeline_configs = pipeline_configs
        self.MAX_POSSIBLE_SEQUENCE_LENGTH = max_seq_len

        if not (10 <= max_seq_len <= 512):
            self.program_halt("max_seq_len should be >= 128 and <= 512. ")

        if not isinstance(arg_model_name, str):
            self.program_halt("invalid model_name. model_name should be string.")

        try:
            self.lp("BACKEND:\t1-Initializing AutoConfig : " + arg_model_name + " ...")
            self.config = AutoConfig.from_pretrained(arg_model_name, output_hidden_states=True)
            self.config.return_dict = True
            self.lp("\tsuccessful.")
            if self.MAX_POSSIBLE_SEQUENCE_LENGTH > self.config.max_position_embeddings:
                msg = ["max_seq_len is greater than config.max_position_embeddings, which is " + str(self.config.max_position_embeddings)]
                msg+= ["setting MAX_POSSIBLE_SEQUENCE_LENGTH to : " + str(self.config.max_position_embeddings)]
                self.lp(msg)
                self.MAX_POSSIBLE_SEQUENCE_LENGTH = self.config.max_position_embeddings

            #self.tokenizer = self.tokenizer
        except Exception as E:
            self.lp(sys.path)
            self.program_halt("Error loading the model. Error: " + str(E))

        try:
            if tokenizer_folder_path is None:
                tokenizer_folder_path = arg_model_name
            self.lp("BACKEND:\t2-Initializing AutoTokenizer from : " + tokenizer_folder_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder_path, config=self.config , use_fast=arg_use_fast_tokenizer)
            self.lp("\tsuccessful.")
            #<<<CRITICAL>>> add additional unused tokens as special tokens...
            if not "[unused1]" in self.tokenizer.vocab:
                self.lp("BACKEND:\t3-Adding special tokens, e.g. [unused1] ... ")
                additional_unused_tokens = ["[unused"+str(i)+"]" for i in range(1,51)]
                self.tokenizer.add_special_tokens({"additional_special_tokens": additional_unused_tokens})
                self.lp("\tsuccessful.")
            else:
                self.lp("BACKEND:\t3-Special tokens, e.g. [unused1] successfully found in the tokenizer ...")
        except Exception as E:
            self.program_halt("Error in AutoTokenizer. Error: " + str(E))

        try:
            self.lp("BACKEND:\t4-Loading pretrained model ... ")
            self.pretrained_model = TFAutoModel.from_pretrained(arg_model_name, config=self.config)
            self.lp("\tsuccessful.")
        except Exception as E:
            self.lp("Error in loading model. Error: " + str(E))
            self.lp("Trying to load with from_pt=True")
            try:
                self.pretrained_model = TFAutoModel.from_pretrained(arg_model_name, config=self.config, from_pt=True)
                self.lp("\tsuccessful.")
            except Exception as E:
                self.program_halt("Error in loading model. Error: " + str(E))

    def tokenize_text(self,text, add_start_end_tokens=False):
        if add_start_end_tokens:
            return [self.tokenizer.cls_token] + self.tokenizer.tokenize(text) + [self.tokenizer.sep_token] #use correct CLS and SEP tokens based on this particular model vocab!
        else:
            return self.tokenizer.tokenize(text)

    def vectorize_without_taking_max_len_into_account(self,sentence_tokens_list):
        """
        This is for CROSS_SENTENCE example and ann_io generation, for the whole document.
        NOTES:
            1) It DOES NOT take into account the bert max_len.
            2) It DOES NOT add [CLS] or [SEP]
        """
        return self.tokenizer.convert_tokens_to_ids(sentence_tokens_list)

    def __create_adam_warmup_optimizer(self, selected_learning_rate, num_train_examples, batch_size, epochs, warmup_proportion=0.1):
        steps_per_epoch = math.ceil(num_train_examples / batch_size)
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = math.floor(num_train_steps * warmup_proportion)

        # Mostly defaults from transformers.optimization_tf
        optimizer, lr_scheduler = create_optimizer(
            selected_learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            min_lr_ratio=0.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay_rate=0.01,
            power=1.0,
        )
        return optimizer

    def build_model(self, selected_optimizer, selected_learning_rate, num_train_examples, selected_mini_batch_size, selected_NoEpochs, warmup_proportion=0.1):
        seq_len = self.MAX_POSSIBLE_SEQUENCE_LENGTH
        Y_dim = self.pipeline_configs['RelationTypeEncoding'].number_of_classes

        L_input_ids = Input(shape=(seq_len,), dtype='int32', name='input_ids')
        L_attention_mask = Input(shape=(seq_len,), dtype='int32', name='attention_mask')
        ANN_inputs = [L_input_ids, L_attention_mask]

        L_pretrained_outputs = self.pretrained_model(ANN_inputs)
        L_CLS = L_pretrained_outputs['last_hidden_state'][:, 0, :]  # CLS

        if self.pipeline_configs['classification_type'] in ['binary' , 'multi-class']:
            L_output = Dense(Y_dim, activation='softmax')(L_CLS)
            loss = CategoricalCrossentropy()
        else:
            L_output = Dense(Y_dim, activation='sigmoid')(L_CLS)
            loss = BinaryCrossentropy()

        ANN_outputs = [L_output]
        self.model = Model(inputs=ANN_inputs, outputs=ANN_outputs)

        if selected_optimizer == "adam":
            myOptimizer = tensorflow.keras.optimizers.Adam(learning_rate=selected_learning_rate)
        elif selected_optimizer == "nadam":
            myOptimizer = tensorflow.keras.optimizers.Nadam(learning_rate=selected_learning_rate)
        elif selected_optimizer == "sgd":
            myOptimizer = tensorflow.keras.optimizersSGD(learning_rate=selected_learning_rate)
        elif selected_optimizer == "adamwarmup":
            myOptimizer = self.__create_adam_warmup_optimizer(selected_learning_rate, num_train_examples, selected_mini_batch_size, selected_NoEpochs, warmup_proportion)

        try:
            self.lp ("Building model ...")
            self.model.compile(optimizer=myOptimizer, loss=loss)
            self.lp ("successful.")
        except Exception as E:
            self.program_halt("Error in building model. Error: " + str(E))
