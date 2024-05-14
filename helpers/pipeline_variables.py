from enum import Enum

class BERT_Representation_Strategy(Enum):
    MARK_FOCUS_ENTITIES = 1
    MASK_EVERYTHING = 2
    MASK_FOCUS_ENTITIES = 3

class BACKEND_Type(Enum):
    KERAS_BERT = 1
    HUGGINGFACE_TENSORFLOW = 2 #TF2.x + a Huggingface model
    HUGGINGFACE_TORCH = 3 #Torch + a Huggingface model + Huggingface Trainer

class MLSetType(Enum):
    TRAINING = 1 # train set
    DEVELOPMENT = 2 # devel set
    TEST = 3 # test set

