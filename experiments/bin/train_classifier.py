import os
from pathlib import Path

import tensorflow as tf
from loguru import logger

from experiments.competitors.glacier.src.help_functions import (
    conditional_pad,
    time_series_normalize,
)
from experiments.data.data import MULTI_DATASETS, UNI_DATASETS, get_data
from experiments.models.classifier import GradientInceptionTimeClassifier

# 0-12
TASK_ID = int(os.getenv("TASK_ID", "-1"))

tf.random.set_seed(1)


def main() -> None:

    names = UNI_DATASETS + MULTI_DATASETS
    data_name = names[TASK_ID]

    out_path = Path(
        f"experiments/out/models/classifier/{data_name}/{tf.__version__}"
    )

    X_train, y_train, X_test, y_test = get_data(data_name)

    logger.info(f"{TASK_ID} - Start dataset: {data_name}")
    try:
        out_path.mkdir(parents=True, exist_ok=True)

        logger.info("Start model without padding")
        model = GradientInceptionTimeClassifier.fit(X_train, y_train)
        y_pred_train = model.predict_cls(X_train)
        y_pred_test = model.predict_cls(X_test)
        logger.info(f"Accuracy on train: {(y_train == y_pred_train).mean()}")
        logger.info(f"Accuracy on test: {(y_test == y_pred_test).mean()}")
        logger.info(f"{TASK_ID} - Saving model")
        model.save(str(out_path / "model_no_padding"))

        if data_name in UNI_DATASETS:
            logger.info("Start glaicer processing")
            _, _, n_timesteps = X_train.shape
            X_train_processed, _ = time_series_normalize(
                data=X_train, n_timesteps=n_timesteps
            )
            X_train_processed_padded, padding_size = conditional_pad(
                X_train_processed
            )
            model = GradientInceptionTimeClassifier.fit(
                X_train_processed_padded, y_train
            )
            logger.info(f"{TASK_ID} - Saving model - no padding")
            model.save(str(out_path / "model_padding"))

            logger.info("Start model with padding (glaicer)")
            logger.info(f"{TASK_ID} - Saving model - padding")
    except Exception as e:
        logger.error(
            f"{TASK_ID} - Error during training on data {data_name}: {e}"
        )
    finally:
        logger.info(f"{TASK_ID} - End dataset: {data_name}")


if __name__ == "__main__":
    main()
