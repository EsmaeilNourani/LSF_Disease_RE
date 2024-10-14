import os
import sys
import argparse
import subprocess

def external_eval_all_entities(pred_folder, program_halt):
    cwd = os.path.dirname(os.path.realpath(__file__))
    execution_folder = os.environ['RT_REL_FOLDERPATH'] or cwd
    if execution_folder[-1] != "/":
        execution_folder += "/"
    devel_gold_folder = execution_folder + "LSD600Corpus/devel-set/" #space is really important at the end

    #<<<CRITICAL>>>: we should use the same command for all experiments
    command = "python3 evalsorel.py --entities lifestyle_factor,disease --relations causes,controls,prevents,treats,statistical_association,positive_statistical_association,negative_statistical_association,no_statistical_association " \
              + devel_gold_folder + " " + pred_folder

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, shell=True)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8').strip()
    stderr = stderr.decode('utf-8').strip()

    f_score = "COULD NOT CALCULATE"
    try:
        f_score = float(stdout.split(" F ")[-1].split("%")[0])
    except Exception as E:
        err_msg = "error in processing external evaluator output:\n"
        err_msg += "cmd: " +  command + "\n"
        err_msg += "stdout: " + stdout + "\n"
        err_msg += "stderr: " + stderr + "\n"
        program_halt(err_msg)
    return f_score, str({"OUT": stdout, "ERR": stderr})

if __name__ == "__main__":
    random_seed_list = [42, 2022, 3240, 13, 8883]
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed_index"            , required=True, type=int , choices=[0,1,2,3,4])
    parser.add_argument("--model_address"                , required=True, type=str)
    parser.add_argument("--train_set_address"            , required=True, type=str)
    parser.add_argument("--devel_set_address"            , required=True, type=str)
    parser.add_argument("--preds_model_output_address" , required=True, type=str)
    parser.add_argument("--logfile_address"              , required=True, type=str)
    args = parser.parse_args()


    # 1: set os.environ and random seed
    args.random_seed = random_seed_list[args.random_seed_index]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  #<<<CRITICAL>>> to turn off the warning
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # 2: set/check folders and logfile
    PARAM_model_folderpath = args.model_address
    PARAM_train_folderpath = args.train_set_address
    PARAM_devel_folderpath = args.devel_set_address
    PARAM_preds_model_output_address = args.preds_model_output_address
    PARAM_logfile_address = args.logfile_address

    if not os.path.isdir(PARAM_model_folderpath):
        print ("invalid path for model_address : " , PARAM_model_folderpath)
        print ("exiting ...")
        sys.exit(-1)

    if not os.path.isdir(PARAM_train_folderpath):
        print ("invalid path for train_set_address : " , PARAM_train_folderpath)
        print ("exiting ...")
        sys.exit(-1)

    if not os.path.isdir(PARAM_devel_folderpath):
        print ("invalid path for devel_set_address : " , PARAM_devel_folderpath)
        print ("exiting ...")
        sys.exit(-1)

    if not os.path.isdir(PARAM_preds_model_output_address):
        print ("invalid path for preds_model_output_address : " , PARAM_preds_model_output_address)
        print ("exiting ...")
        sys.exit(-1)

    # 3: delete log file if exists
    if os.path.isfile(PARAM_logfile_address):
        print("deleting previously existed log_file and recreating: " + PARAM_logfile_address)
        os.remove(PARAM_logfile_address)

    # 4: set paths for import folders
    current_file_path = dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("/".join(current_file_path.split("/")[:-1]))
    sys.path.append("/".join(current_file_path.split("/")[:-2]))
    # ------------------------------------------------------------------------------------------------------------------

    #5: read and set parameters #<<<CRITICAL>>> hardcoded- based on the best found values during optimization (These are also mentioned in the paper).
    args.optimizer = "adam"
    args.num_train_epochs = "75"
    args.minibatch_size = "16"
    args.learning_rate = "5e-6"
    args.max_seq_len = "180"
    from helpers import pipeline_variables
    args.representation_strategy = pipeline_variables.BERT_Representation_Strategy.MASK_EVERYTHING #<<<CRITICAL>>>

    # 6: create training params
    training_arguments = {
            "optimizer"        : args.optimizer,
            "num_train_epochs" : int(args.num_train_epochs),
            "minibatch_size"   : int(args.minibatch_size),
            "learning_rate"    : float(args.learning_rate),
            "output_dir"       : PARAM_preds_model_output_address,
    }
    training_arguments['adam_epsilon'] = 1e-07
    all_params = {
        "cmd_line" : args._get_kwargs(),
        "training" : [x for x in training_arguments.items()] ,
    }

    # 7: create project and pipeline_pt objects ...
    import project
    import relation_extraction_pipeline_pt

    prj = project.Project(PARAM_logfile_address, "LSF_DIS_rel_configs.json")
    prj.lp("PARAMS:\t" + str(all_params))
    pipeline = relation_extraction_pipeline_pt.RelationExtractionPipeline(
        project=prj,
        pretrained_model_name_or_path=PARAM_model_folderpath,
        max_seq_len=int(args.max_seq_len),
        random_seed=args.random_seed,
        evaluation_metric="positive_f1_score",
        external_evaluator=external_eval_all_entities,
        predict_devel=True,
        evaluate_devel=True,
        writeback_devel_preds=False,
        writeback_devel_preds_folder=PARAM_preds_model_output_address,
        process_devel_after_epoch=0,
        representation_strategy=args.representation_strategy,
        save_best_model_folder_path=PARAM_preds_model_output_address
    )

    # 8: add training and devel files -----------------------------------------------
    pipeline.add_train_dev_test_brat_folders(PARAM_train_folderpath, PARAM_devel_folderpath)

    # 9: run pipeline
    pipeline.run_pipeline(training_arguments)

    # 10: safe exit
    prj.exit()
