import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import random
import ssl

import numpy as np
import seaborn as sns
import shortuuid
import tensorflow as tf
import torch
from aeon.classification.deep_learning import InceptionTimeClassifier

import wandb
from classifier.inception import GradientInceptionTimeClassifier
from nte.experiment.default_args0 import parse_arguments
from nte.experiment.utils import (
    backgroud_data_configuration,
    dataset_mapper,
    get_model,
    get_run_configuration,
    number_to_dataset,
    set_global_seed,
)
from nte.models.saliency_model.counterfactual_multi_nomal import CFExplainer
from nte.utils import CustomJsonEncoder

sns.set_style("darkgrid")

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

ENABLE_WANDB = True
WANDB_DRY_RUN = False


BASE_SAVE_DIR = "results_v1/2312/"
if WANDB_DRY_RUN:
    os.environ["WANDB_MODE"] = "dryrun"

if __name__ == "__main__":
    args = parse_arguments()
    print("Config: \n", json.dumps(args.__dict__, indent=2))

    if args.dataset in number_to_dataset.keys():
        args.dataset = number_to_dataset[args.dataset]

    if args.enable_seed:
        set_global_seed(args.seed_value)

    ENABLE_SAVE_PERTURBATIONS = args.save_perturbations
    PROJECT_NAME = args.pname
    dataset = dataset_mapper(DATASET=args.dataset)

    X_train, y_train = dataset.load_train_data()
    X_test, y_test = dataset.load_test_data()

    X_train, y_train = X_train, y_train
    X_test, y_test = X_test, y_test

    model = InceptionTimeClassifier(n_epochs=20, batch_size=4)
    model.fit(X_train, y_train)

    model = GradientInceptionTimeClassifier(model)

    TAG = f"{args.algo}-{args.dataset}-{args.background_data}-{args.background_data_perc}-run-{args.run_id}"
    BASE_SAVE_DIR = BASE_SAVE_DIR + "/" + TAG

    # model = get_model(dataset=args.dataset, input_size=6, num_classes=4)

    # softmax_fn = torch.nn.Softmax(dim=-1)

    bg_data, bg_label, bg_len = backgroud_data_configuration(
        BACKGROUND_DATA=args.background_data,
        BACKGROUND_DATA_PERC=args.background_data_perc,
        dataset=dataset,
    )

    print(
        f"Using {args.background_data_perc}% of background data. Samples: {bg_len}"
    )

    config = args.__dict__

    explainer = None

    if args.algo == "cf":
        explainer = CFExplainer(
            background_data=bg_data[:bg_len],
            background_label=bg_label[:bg_len],
            predict_fn=model,
            enable_wandb=ENABLE_WANDB,
            args=args,
            use_cuda=False,
        )

    config = {
        **config,
        **{
            "tag": TAG,
            "algo": args.algo,
        },
    }

    dataset_len = len(dataset.test_data)

    ds = get_run_configuration(
        args=args, dataset=dataset, TASK_ID=args.task_id
    )

    res_path = f"mcels/{args.dataset}/"
    os.system(f'mkdir -p "{res_path}"')

    cf_res = []
    cf_probs = []
    cf_maps = []
    for ind, (
        original_signal,
        original_label,
    ) in ds:  # original_signal is from the test set
        try:
            if args.enable_seed_per_instance:
                set_global_seed(random.randint())
            metrics = {"epoch": {}}
            cur_ind = (
                args.single_sample_id
                if args.run_mode == "single"
                else (ind + (int(args.task_id) * args.samples_per_task))
            )
            UUID = (
                dataset.valid_name[cur_ind]
                if args.dataset_type == "valid"
                else shortuuid.uuid()
            )
            EXPERIMENT_NAME = f"{args.algo}-{cur_ind}-R{args.run_id}-{UUID}-C{ind}-T{args.task_id}-S{args.samples_per_task}-TS{(int(args.task_id) * args.samples_per_task)}-TT{(ind+int(args.task_id) * args.samples_per_task)}"
            print(
                f" {args.algo}: Working on dataset: {args.dataset} index: {cur_ind} [{((cur_ind + 1) / dataset_len * 100):.2f}% Done]"
            )
            SAVE_DIR = f"{BASE_SAVE_DIR}/{EXPERIMENT_NAME}"
            os.system(f'mkdir -p "{SAVE_DIR}"')
            os.system(f'mkdir -p "./wandb/{TAG}/"')
            config["save_dir"] = SAVE_DIR

            if args.run_mode == "single":
                config = {**config}

            json.dump(
                config,
                open(SAVE_DIR + "/config.json", "w"),
                indent=2,
                cls=CustomJsonEncoder,
            )
            if ENABLE_WANDB:
                wandb.init(
                    project=PROJECT_NAME,
                    name=EXPERIMENT_NAME,
                    tags=TAG,
                    config=config,
                    reinit=True,
                    force=True,
                    dir=f"./wandb/{TAG}/",
                )

            # original_signal = torch.tensor(
            #     original_signal, dtype=torch.float32
            # )  # original_signal ->ntest instance

            # original_signal = torch.tensor(
            #     original_signal, dtype=torch.float32
            # )

            with torch.no_grad():
                if args.bbm == "dnn":
                    print(original_signal.shape)
                    target = model(
                        original_signal.reshape(
                            1,
                            original_signal.shape[0],
                            original_signal.shape[1],
                        )
                    )

                else:
                    raise Exception(
                        f"Black Box model not supported: {args.bbm}"
                    )
            # print("target", target)

            category = np.argmax(target)  # prediction label
            args.dataset = dataset
            if ENABLE_WANDB:
                wandb.run.summary[f"ori_prediction_class"] = category
                wandb.run.summary[f"ori_prediction_prob"] = np.max(target)
                wandb.run.summary[f"ori_label"] = original_label

            """
            During the execution of a run or experiment, you can use wandb.run.summary to store and update
            important metrics, statistics, or any other relevant information that you want to track and analyze. 
            The wandb.run.summary dictionary is automatically synchronized with the Weights & Biases server, 
            allowing you to view and compare the summary information across multiple runs.
            """

            if args.background_data == "none":
                explainer.background_data = original_signal
                explainer.background_label = original_label

            mask, ori_perturbated, target_prob = explainer.generate_saliency(
                data=original_signal,
                label=original_label,
                save_dir=SAVE_DIR,
                target=target,
                dataset=dataset,
            )

            cf = ori_perturbated.flatten()  # Convert tensor to NumPy array
            cf_res.append(cf)

            # perturbation_res = torch.tensor(
            #     ori_perturbated, dtype=torch.float32
            # )

            perturbation_res = ori_perturbated

            # print("perturbation_output shape :", perturbation_output.shape)  (1, ts_length)
            pert_res = model(
                perturbation_res.reshape(
                    1, perturbation_res.shape[0], perturbation_res.shape[1]
                )
            )

            pert_label = np.argmax(pert_res)  # prediction label

            cf_probs.append(target_prob)

            cf_maps.append(mask)

            if ENABLE_WANDB:
                wandb.run.summary[f"pert_prediction_class"] = pert_label
                wandb.run.summary[f"target_prob"] = target_prob
                wandb.run.summary[f"mask"] = mask

        except Exception as e:
            with open(f"/tmp/{TAG}_error.log", "a+") as f:
                f.write(e)
                f.write(e.__str__())
                f.write("\n\n")

    np.save(res_path + "saliency_cf.npy", np.array(cf_res))
    np.save(res_path + "saliency_cf_prob.npy", np.array(cf_probs))
    np.save(res_path + "map_cf.npy", np.array(cf_maps))
