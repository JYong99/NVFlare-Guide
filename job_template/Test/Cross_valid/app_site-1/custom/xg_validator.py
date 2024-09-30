import time
import logging, time, json

import numpy as np, xgboost as xgb

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception

from sklearn.metrics import roc_auc_score
from nvflare.app_opt.xgboost.tree_based.shareable_generator import update_model

def evaluate_model(x_test, model, y_test):
    # Make predictions on the testing set
    dtest = xgb.DMatrix(x_test)
    y_pred = model.predict(dtest)

    # Evaluate the model
    auc = roc_auc_score(y_test, y_pred)
    return auc

class XG_Validator(Executor):
    def __init__(
        self,
        sleep_time=0,
        validate_task_name=AppConstants.TASK_VALIDATION,
    ):
        super().__init__()
        self.logger = logging.getLogger("XGValidator")
        self._validate_task_name = validate_task_name
        self._sleep_time = sleep_time

        #Check how to validate?
        #Load dataset, AUC, return
        #Make example from np_trainer/ np_validator
    
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
                self.log_info(
                    fl_ctx, f"The shareable is: {shareable}"
                )
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
            
            test = xg_model.get("model_data")
            test2 = json.loads(test)
            # The workflow provides MODEL_OWNER information in the shareable header.
            model_name = shareable.get_header(AppConstants.MODEL_OWNER, "?")
            loadable_model = bytearray(json.dumps(test2), "utf-8")
            model.load_model(loadable_model)
            # Print properties.
            self.log_info(fl_ctx, f"Model: \n{xg_model}")
            self.log_info(fl_ctx, f"Task name: {task_name}")
            self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
            self.log_info(fl_ctx, f"Validating model from {model_name}.")

            # Check abort signal regularly.
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            
            self.log_info(fl_ctx, "Loaded Model <---------")
            #self.log_info(fl_ctx, f"Adding  AUC validation.")
            # val_results = {}
            
            # auc = evaluate_model(x_test, model, y_test)

            

        