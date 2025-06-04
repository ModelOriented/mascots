from loguru import logger

from bin.utils.data import get_data
from bin.utils.device import DEVICE
from bin.utils.models import LSTM, get_default_params

logger.info("Load data")
X_train, y_train, X_test, y_test = get_data("FaultDetectionA")

logger.info("Init model")
lstm = LSTM(
    X_train.shape[2], 3, hidden_size=1024, num_layers=2, weight_decay=0.0
).to(DEVICE)

logger.info("Train model")
params = get_default_params()
params["max_epochs"] = 10
lstm.fit(X_train, y_train, X_test, y_test, params)
