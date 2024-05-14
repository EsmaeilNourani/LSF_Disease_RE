import os
import sys

#Code to add the module paths to python
current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))
#-------------------------------------------------------------------------

class BERT_helper:
    def __init__(self, lp, program_halt, bert_model_folder_address, max_seq_len=512):
        self.lp = lp
        self.program_halt = program_halt
        self.MAX_POSSIBLE_SEQUENCE_LENGTH = max_seq_len

        if not (10 <= max_seq_len <= 512):
            self.program_halt("max_seq_len should be >= 128 and <= 512. ")

        if bert_model_folder_address[-1] != "/":
            bert_model_folder_address += "/"

        self.bert_model_folder_address = bert_model_folder_address

        try:
            if os.path.isfile(self.bert_model_folder_address + "vocab.txt"):
                vocab_file_name = "vocab.txt"
            else:
                vocab_file_name = "vocab_cased_pubmed_pmc_30k.txt"

            self.lp ("Initializing vanilla bert tokenizer with :" + self.bert_model_folder_address + vocab_file_name)
            import BioDeepRel.bert.tokenization as tokenization
            self.__tokenizer = tokenization.FullTokenizer(self.bert_model_folder_address + vocab_file_name, do_lower_case=False)
            self.lp ("PASS.")
        except Exception as E:
            self.lp (sys.path)
            self.program_halt("Error bert tokenizer :" + str(E))

    def tokenize_text(self,text, add_start_end_tokens=False):
        if add_start_end_tokens:
            return ["[CLS]"] + self.__tokenizer.tokenize(text) + ["[SEP]"]
        else:
            return self.__tokenizer.tokenize(text)

    def tokenize_tokenList (self, tokens_list, add_start_end_tokens=False):
        results = []
        for token in tokens_list:
            results.extend(self.__tokenizer.tokenize(token))

        if add_start_end_tokens:
            return ["[CLS]"] + results + ["[SEP]"]
        else:
            return results

    def vectorize(self,sentence_tokens_list, max_seq_len, add_start_end_tokens=True):
        """
        Function is decidedly written to be more clear. It could have been written more compactly.
        """
        if add_start_end_tokens:
            """
            Then, we know that the max_seq_len has been calculated without considering [CLS] and end [SEP] tokens.
            Hence, we add 2 to that for correct number. 
            """
            max_seq_len+=2
            if max_seq_len > self.MAX_POSSIBLE_SEQUENCE_LENGTH:
                max_seq_len = self.MAX_POSSIBLE_SEQUENCE_LENGTH
            sentence_tokens_list = ["[CLS]"] + sentence_tokens_list [:max_seq_len-2] + ["[SEP]"]
            return [self.__tokenizer.vocab[token] for token in sentence_tokens_list]

        else:
            """
            Then, we know that max_seq_len is calculated considering start and end tokens.
            """
            if max_seq_len > self.MAX_POSSIBLE_SEQUENCE_LENGTH:
                max_seq_len = self.MAX_POSSIBLE_SEQUENCE_LENGTH
            sentence_tokens_list = sentence_tokens_list[1:-1] #remove first and last tokens ("[CLS]" and "[SEP]")
            sentence_tokens_list = ["[CLS]"] + sentence_tokens_list[:max_seq_len-2] + ["[SEP]"]
            return [self.__tokenizer.vocab[token] for token in sentence_tokens_list]

    def vectorize_without_taking_max_len_into_account(self,sentence_tokens_list):
        """
        This is for CROSS_SENTENCE example and ann_io generation, for the whole document.
        NOTES:
            1) It DOES NOT take into account the bert max_len.
            2) It DOES NOT add [CLS] or [SEP]
        """
        return [self.__tokenizer.vocab[token] for token in sentence_tokens_list]

    def build_vanilla_BERT_model(self, max_seq_len):
        from helpers import general_helpers as gf

        #config json
        config_files_list = gf.get_all_files_with_extension(self.bert_model_folder_address, "json")
        if len(config_files_list) != 1:
            self.program_halt("there should be exactly one .json file in " + self.bert_model_folder_address)
        fileaddress_config  = config_files_list[0]

        #ckpt
        checkpoint_files_list = gf.get_all_files_with_extension(self.bert_model_folder_address, "meta")
        if len(checkpoint_files_list) !=1 :
            self.program_halt("there should be exactly one .meta file in " + self.bert_model_folder_address)
        fileaddress_checkpoint = checkpoint_files_list[0].split(".meta")[0]

        #<<<CRITICAL>>> See DeepRelExtratorPipeline.__GenerateANNMatrices
        max_seq_len += 2
        if max_seq_len > self.MAX_POSSIBLE_SEQUENCE_LENGTH:
            max_seq_len = self.MAX_POSSIBLE_SEQUENCE_LENGTH

        try:
            import keras_bert
            model = keras_bert.load_trained_model_from_checkpoint(fileaddress_config, fileaddress_checkpoint, training=False, trainable=True, seq_len=max_seq_len)
            return model
        except Exception as E:
            self.program_halt ("Error in loading bioBERT model. Error: " + str(E))
