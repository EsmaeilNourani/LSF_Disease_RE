import os
import sys
import argparse

current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_file_path.split("/")[:-1]))
sys.path.append("/".join(current_file_path.split("/")[:-2]))

import use_prediction_pipeline_pt

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_file_path" , required=True, type=str)
    parser.add_argument("--log_file_path" , required=True, type=str)
    parser.add_argument("--model_folder_path" , required=True, type=str)
    parser.add_argument("--input_folder_path" , required=True, type=str)
    parser.add_argument("--output_folder_path" , required=True, type=str)
    parser.add_argument("--create_output_ann_files" , type=bool, default=True)
    parser.add_argument("--dont_generate_negatives_if_sentence_distance_ge" , type=int , default=7)
    args = parser.parse_args()

    mypipeline = use_prediction_pipeline_pt.LargeScalePredictionPipeline_torch(
        configs_file_path = args.configs_file_path,
        pretrained_model_path = args.model_folder_path,
        log_file_path = args.log_file_path,
        input_folder_path = args.input_folder_path,
        output_folder_path = args.output_folder_path,
        create_output_ann_files=args.create_output_ann_files,
        dont_generate_negatives_if_sentence_distance_ge = args.dont_generate_negatives_if_sentence_distance_ge,
        write_logits_to_output_tsv=False,
    )

    mypipeline.run_large_scale_pipeline()
