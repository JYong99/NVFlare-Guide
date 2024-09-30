import logging
import time

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception
from nvflare.app_opt.xgboost.tree_based.shareable_generator import update_model

from .constants import NPConstants


def to_dataset_tuple(data: dict):
    dataset_tuples = {}
    for dataset_name, dataset in data.items():
        dataset_tuples[dataset_name] = _to_data_tuple(dataset)
    return dataset_tuples


def _to_data_tuple(data):
    data_num = data.shape[0]
    # split to feature and label
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x.to_numpy(), y.to_numpy(), data_num


def load_features(feature_data_path: str) -> List:
    try:
        features = []
        with open(feature_data_path, "r") as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            line_list = next(csv_reader)
            features = line_list
        return features
    except Exception as e:
        raise Exception(f"Load header for path'{feature_data_path} failed! {e}")


def load_data(
    data_path: str, data_features: List, random_state: int, test_size: float, skip_rows=None
) -> Dict[str, pd.DataFrame]:
    try:
        df: pd.DataFrame = pd.read_csv(
            data_path, names=data_features, sep=r"\s*,\s*", engine="python", na_values="?", skiprows=skip_rows
        )

        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

        return {"train": train, "test": test}

    except Exception as e:
        raise Exception(f"Load data for path '{data_path}' failed! {e}")


def transform_data(data: Dict[str, Tuple]) -> Dict[str, Tuple]:
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaled_datasets = {}
    for dataset_name, (x_data, y_data, data_num) in data.items():
        x_scaled = scaler.fit_transform(x_data)
        scaled_datasets[dataset_name] = (x_scaled, y_data, data_num)
    return scaled_datasets

def evaluate_model(x_test, model, y_test):
    # Make predictions on the testing set
    dtest = xgb.DMatrix(x_test)
    y_pred = model.predict(dtest)

    # Evaluate the model
    auc = roc_auc_score(y_test, y_pred)
    return auc



class NPValidator(Executor):
    def __init__(
        self,
        epsilon=1,
        sleep_time=0,data_root_dir,
        validate_task_name=AppConstants.TASK_VALIDATION,
    ):
        # Init functions of components should be very minimal. Init
        # is called when json is read. A big init will cause json loading to halt
        # for long time.
        super().__init__()

        self.logger = logging.getLogger("NPValidator")
        self._random_epsilon = epsilon
        self._sleep_time = sleep_time
        self._validate_task_name = validate_task_name
        self._data_root_dir = data_root_dir

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # if event_type == EventType.START_RUN:
        #     Create all major components here. This is a simple app that doesn't need any components.
        # elif event_type == EventType.END_RUN:
        #     # Clean up resources (closing files, joining threads, removing dirs etc.)
        pass

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Any long tasks should check abort_signal regularly.
        # Otherwise, abort client will not work.
        site_name = fl_ctx.get_site_name()
        feature_data_path = f"{self._data_root_dir}/{site_name}_header.csv"
        features = load_features(feature_data_path)
        n_features = len(features) - 1  # remove label

        data_path = f"{self._data_root_dir}/{site_name}.csv"
        data = load_data(
            data_path=data_path, data_features=features, random_state=0, test_size=test_size, skip_rows=None
        )

        data = to_dataset_tuple(data)
        dataset = transform_data(data)
        x_train, y_train, train_size = dataset["train"]
        x_test, y_test, test_size = dataset["test"]

        xgb_params = {
            "eta": 0.1 / num_client_bagging,
            "objective": "binary:logistic",
            "max_depth": 8,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1.0,
            "tree_method": "hist",
        }
        model = xgb.xgb.Booster()
        config = model.save_config()

        global_model_as_dict = None

        if task_name == self._validate_task_name:
            try:
                # First we extract DXO from the shareable.
                try:
                    model_dxo = from_shareable(shareable)
                except Exception as e:
                    self.log_error(
                        fl_ctx, f"Unable to extract model dxo from shareable. Exception: {secure_format_exception(e)}"
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Get model from shareable. data_kind must be WEIGHTS.
                if model_dxo.data and model_dxo.data_kind == DataKind.WEIGHTS:
                    model_data = model_dxo.data
                else:
                    self.log_error(
                        fl_ctx, "Model DXO doesn't have data or is not of type DataKind.WEIGHTS. Unable to validate."
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)
                for update in model_data:
                    global_model_as_dict = update_model(global_model_as_dict, json.loads(update))
                loadable_model = bytearray(json.dumps(global_model_as_dict), "utf-8")
                model.load_model(loadable_model)
                model.load_config(config)

                # Check if key exists in model
                #if NPConstants.NUMPY_KEY not in model:
                #    self.log_error(fl_ctx, "numpy_key not in model. Unable to validate.")
                #    return make_reply(ReturnCode.BAD_TASK_DATA)

                self.log_info(fl_ctx, f"Adding  AUC validation.")
                val_results = {}

                auc = evaluate_model(x_test, model, y_test) ###

                val_results["accuracy"] = auc

                # Check abort signal regularly.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"Validation result: {val_results}")

                # Create DXO for metrics and return shareable.
                metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
                return metric_dxo.to_shareable()
            except Exception as e:
                self.log_exception(fl_ctx, f"Exception in NPValidator execute: {secure_format_exception(e)}.")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)