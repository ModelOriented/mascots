from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import wandb
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from tslearn.neighbors import KNeighborsTimeSeries

from experiments.competitors.mcels.nte.experiment.utils import (
    save_timeseries_mul,
    tv_norm,
)
from experiments.competitors.mcels.nte.models.saliency_model import Saliency


class CFExplainer(Saliency):
    def __init__(
        self,
        background_data,
        background_label,
        predict_fn,
        enable_wandb,
        use_cuda,
        args,
    ):
        super(CFExplainer, self).__init__(
            background_data=background_data,
            background_label=background_label,
            predict_fn=predict_fn,
        )
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.conf_threshold = 0.8
        self.eps = None
        self.eps_decay = 0.9991

    def native_guide_retrieval(
        self, query, target_label, distance, n_neighbors
    ):
        dim_nums, ts_length = query.shape[0], query.shape[1]
        df = pd.DataFrame(self.background_label, columns=["label"])

        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
        knn.fit(
            self.background_data[
                list(df[df["label"] == target_label].index.values)
            ]
        )

        dist, ind = knn.kneighbors(
            query.reshape(1, dim_nums, ts_length), return_distance=True
        )
        return dist, df[df["label"] == target_label].index[ind[0][:]]

    def cf_label_fun(self, instance):
        print(f"{instance.shape=} HERE")
        # raise Exception
        output = self.predict_fn(instance)
        print(f"{output.shape=} HERE")

        target = tf.argsort(output, direction="DESCENDING", axis=0)[0, 1]
        return target.numpy()

    def generate_saliency(self, data, label, **kwargs):
        self.mode = "Explore"
        query = data.copy()

        if isinstance(data, np.ndarray):
            data = tf.convert_to_tensor(data, dtype=tf.float32)

        top_prediction_class = np.argmax(kwargs["target"])

        cf_label = self.cf_label_fun(data)

        dis, idx = self.native_guide_retrieval(query, cf_label, "euclidean", 1)
        NUN = self.background_data[idx.item()]

        self.eps = 1.0

        mask_init = np.random.uniform(size=data.shape, low=0, high=1)
        mask = tf.Variable(mask_init, dtype=tf.float32, trainable=True)

        # Setup optimizer
        optimizer = tf.optimizers.Adam(learning_rate=self.args.lr)

        if self.args.enable_lr_decay:
            # For learning rate decay in TensorFlow
            scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.args.lr,
                decay_steps=100000,
                decay_rate=self.args.lr_decay,
            )
            optimizer.learning_rate = scheduler

        print(f"{self.args.algo}: Optimizing... ")
        metrics = defaultdict(lambda: [])

        max_iterations_without_improvement = (
            100  # Define the maximum number of iterations without improvement
        )
        imp_threshold = 0.001
        best_loss = float("inf")  # Track the best 'Confidence' achieved
        counter = 0  # Counter for iterations without improvement

        # Training
        i = 0
        while i <= self.args.max_itr:
            Rt = tf.convert_to_tensor(NUN, dtype=tf.float32)

            perturbated_input = data * (1 - mask) + Rt * mask

            pred_outputs = self.predict_fn(
                tf.reshape(
                    perturbated_input,
                    [
                        1,
                        perturbated_input.shape[0],
                        perturbated_input.shape[1],
                    ],
                ).numpy()
            )

            l_maximize = 1 - pred_outputs[0][cf_label]
            l_budget_loss = tf.reduce_mean(tf.abs(mask)) * float(
                self.args.enable_budget
            )
            l_tv_norm_loss = tv_norm(mask, self.args.tv_beta) * float(
                self.args.enable_tvnorm
            )

            loss = (
                (self.args.l_budget_coeff * l_budget_loss)
                + (self.args.l_tv_norm_coeff * l_tv_norm_loss)
                + (self.args.l_max_coeff * l_maximize)
            )

            if best_loss - loss < imp_threshold:
                counter += 1
            else:
                counter = 0  # Reset counter if there is an improvement
                best_loss = loss  # Update the best 'Confidence' achieved

            # gradients computation
            with tf.GradientTape() as tape:
                tape.watch(mask)
                perturbated_input = data * (1 - mask) + Rt * mask
                pred_outputs = self.predict_fn(
                    tf.reshape(
                        perturbated_input,
                        [
                            1,
                            perturbated_input.shape[0],
                            perturbated_input.shape[1],
                        ],
                    ).numpy()
                )
                l_maximize = 1 - pred_outputs[0][cf_label]
                l_budget_loss = tf.reduce_mean(tf.abs(mask)) * float(
                    self.args.enable_budget
                )
                l_tv_norm_loss = tv_norm(mask, self.args.tv_beta) * float(
                    self.args.enable_tvnorm
                )

                loss = (
                    (self.args.l_budget_coeff * l_budget_loss)
                    + (self.args.l_tv_norm_coeff * l_tv_norm_loss)
                    + (self.args.l_max_coeff * l_maximize)
                )

            gradients = tape.gradient(loss, [mask])
            optimizer.apply_gradients(zip(gradients, [mask]))

            metrics["L_Maximize"].append(float(l_maximize.numpy()))
            metrics["L_Budget"].append(float(l_budget_loss.numpy()))
            metrics["L_TV_Norm"].append(float(l_tv_norm_loss.numpy()))
            metrics["L_Total"].append(float(loss.numpy()))
            metrics["CF_Prob"].append(float(pred_outputs[0][cf_label]))

            if self.args.enable_lr_decay:
                scheduler.apply(optimizer, epoch=i)

            # Clamp mask
            mask.assign(tf.clip_by_value(mask, 0, 1))

            if self.enable_wandb:
                _mets = {
                    **{k: v[-1] for k, v in metrics.items() if k != "epoch"}
                }
                if f"epoch_{i}" in metrics["epoch"]:
                    _mets = {
                        **_mets,
                        **metrics["epoch"][f"epoch_{i}"]["eval_metrics"],
                    }
                wandb.log(_mets)

            # Check if early stopping condition is met
            if counter >= max_iterations_without_improvement:
                print(
                    "Early stopping triggered: 'total loss' metric didn't improve much"
                )
                break
            else:
                i += 1

        mask = mask.numpy()

        threshold = 0.5
        converted_mask = np.where(mask >= threshold, mask, 0)
        Rt = tf.convert_to_tensor(NUN, dtype=tf.float32)
        converted_mask = tf.convert_to_tensor(converted_mask, dtype=tf.float32)
        perturbated_input = data * (1 - converted_mask) + Rt * converted_mask

        pred_outputs = self.predict_fn(
            tf.reshape(
                perturbated_input,
                [1, perturbated_input.shape[0], perturbated_input.shape[1]],
            ).numpy()
        )

        target_prob = float(pred_outputs[0][cf_label])

        converted_mask = converted_mask.numpy().flatten()

        save_timeseries_mul(
            mask=converted_mask,
            raw_mask=None,
            time_series=data.numpy(),
            perturbated_output=perturbated_input.numpy(),
            save_dir=kwargs["save_dir"],
            enable_wandb=self.enable_wandb,
            algo=self.args.algo,
            dataset=self.args.dataset,
            category=top_prediction_class,
        )

        perturbated_input = perturbated_input.numpy()

        return converted_mask, perturbated_input, target_prob
