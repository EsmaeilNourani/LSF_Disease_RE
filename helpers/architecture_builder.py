import os
import sys
#Code to add the module paths to python
current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))


class Architecture_Builder(object):
    def __init__(self, lp, program_halt, configs , BERTUtils):
        self.lp = lp
        self.program_halt = program_halt
        self.configs = configs
        self.BERTUtils = BERTUtils

        if (self.BERTUtils is None):
            self.program_halt ("Cannot load model. No BERT checkpoint-folder is defined in the config file.")
            sys.exit(-1)

        if not self.configs['classification_type'] in ['binary' , 'multi-class' , 'multi-label']:
            self.program_halt("invalid classification type in the config file: " + str(self.configs['classification_type']))

        if self.configs['classification_type'] in ['binary' , 'multi-class']:
            self.decision_layer_output_funtion_name = "softmax"
        else:
            self.decision_layer_output_funtion_name = "sigmoid" #multi-label

    def generate_BERT_pair_prediction(self, model_params):
        from keras.models import Model
        import keras.layers
        import keras.layers.core
        import keras.backend as K

        #hyper-parameters
        H_BERT_dropout_rate = model_params['bertdr']
        H_dense_out_dim     = model_params['dod']
        H_dropout_rate_after_dense = model_params['drad']
        H_dense_actv_func = "tanh"
        should_consider = lambda x: (x is not None) and (isinstance(x, int) or isinstance(x, float)) and (x > 0) #function to evaluate integer values

        #Model inputs/outputs
        MODEL_Outputs = []

        #creating model
        bioBERT_model = self.BERTUtils.build_vanilla_BERT_model (self.BERTUtils.MAX_POSSIBLE_SEQUENCE_LENGTH)
        BERT_OUT = keras.layers.core.Lambda(lambda x: K.slice(x, [0, 0, 0], [-1, 1, -1]), name="BERT_OUT_SLICE")(bioBERT_model.layers[-1].output)
        BERT_OUT = keras.layers.core.Flatten(name="BERT_OUT_FLAT")(BERT_OUT)

        MODEL_Inputs = []
        MODEL_Inputs.extend(bioBERT_model.input)

        # add dropout to BERT output
        if should_consider(H_BERT_dropout_rate):
            BERT_OUT = keras.layers.core.Dropout(H_BERT_dropout_rate , name="BERT_OUT_Dropout")(BERT_OUT)

        L_all_features = BERT_OUT

        # Dense/dropout/decision layers
        Y_dim = self.configs['RelationTypeEncoding'].number_of_classes

        if should_consider(H_dense_out_dim):
            L_dense = keras.layers.core.Dense(units=H_dense_out_dim, activation=H_dense_actv_func, trainable=True)(L_all_features)
            if should_consider(H_dropout_rate_after_dense):
                L_dense = keras.layers.core.Dropout(H_dropout_rate_after_dense)(L_dense)
            L_decision = keras.layers.core.Dense(units=Y_dim, activation=self.decision_layer_output_funtion_name, name="decision_Y", trainable=True)(L_dense)
        else:
            L_decision = keras.layers.core.Dense(units=Y_dim, activation=self.decision_layer_output_funtion_name, name="decision_Y", trainable=True)(L_all_features)

        MODEL_Outputs.append(L_decision)
        model = Model(inputs=MODEL_Inputs, outputs=MODEL_Outputs)
        return model
