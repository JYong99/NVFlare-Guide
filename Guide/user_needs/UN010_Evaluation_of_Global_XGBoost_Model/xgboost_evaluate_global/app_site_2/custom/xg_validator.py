import logging, time, json

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception

import xgboost as xgb, pandas as pd
from sklearn.metrics import roc_auc_score


def Validate(site_name, model):
    df = pd.read_csv(site_name)
    
    X = df.drop(df.columns[0], axis=1)
    y = df[df.columns[0]]
    dtest = xgb.DMatrix(data=X)

    predictions = model.predict(dtest)

    return roc_auc_score(y, predictions)


class XG_Validator(Executor):
    def __init__(
        self,
        csv_dir = "",
        sleep_time=0,
        validate_task_name=AppConstants.TASK_VALIDATION,
    ):
        super().__init__()
        self.logger = logging.getLogger("XGValidator")
        self.csv_dir = csv_dir
        self._validate_task_name = validate_task_name
        self._sleep_time = sleep_time

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        #Checks for any abort signal requested
        count, interval = 0, 0.5
        while count < self._sleep_time:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            time.sleep(interval)
            count += interval

        #If task is validate
        if task_name == self._validate_task_name:
            
            model = xgb.Booster()
            #Extract DXO from shareable
            try:
                model_dxo = from_shareable(shareable)
            except Exception as e:
                self.log_error(
                    fl_ctx, f"Unable to extract model dxo from shareable(Checkpoint 1). Exception: {secure_format_exception(e)}"
                )
                return make_reply(ReturnCode.BAD_TASK_DATA)
            
            # Get model from shareable. data_kind must be WEIGHTS.
            if model_dxo.data and model_dxo.data_kind == DataKind.WEIGHTS:
                xg_model = model_dxo.data
            else:
                self.log_error(
                    fl_ctx, "Model DXO doesn't have data or is not of type DataKind.WEIGHTS. Unable to validate (Checkpoint 2)."
                )
                return make_reply(ReturnCode.BAD_TASK_DATA)
            
            model_data = xg_model.get("model_data")
            dump = json.loads(model_data)
            # The workflow provides MODEL_OWNER information in the shareable header.
            loadable_model = bytearray(json.dumps(dump), "utf-8")
            model.load_model(loadable_model)
            # Print properties.
            self.log_info(fl_ctx, f"Model: \n{xg_model}")
            self.log_info(fl_ctx, f"Task name: {task_name}")
            self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

            # Check abort signal regularly.
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            
            val_results = {}
            
            auc = Validate(self.csv_dir, model)
            val_results[f"accuracy"] = auc
            metric = DXO(data_kind=DataKind.METRICS, data = val_results)
            return metric.to_shareable()
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

            

        