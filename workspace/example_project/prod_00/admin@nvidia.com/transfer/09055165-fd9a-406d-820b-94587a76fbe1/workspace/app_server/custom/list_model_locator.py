from typing import Dict,List, Union
import os, json
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.security.logging import secure_format_exception


class ListModelLocator(ModelLocator):
    SERVER_MODEL_NAME = "server"

    def __init__(self, model_dir="", model_name: Union[str, Dict[str, str]] = "xgboost_model.json"):
    
        super().__init__()

        self.model_dir = model_dir
        if model_name is None:
            self.model_file_name = {ListModelLocator.SERVER_MODEL_NAME: "xgboost_model.json"}
        elif isinstance(model_name, str):
            self.model_file_name = {ListModelLocator.SERVER_MODEL_NAME: model_name}
        elif isinstance(model_name, dict):
            self.model_file_name = model_name
        else:
            raise ValueError(f"model_name must be a str, or a Dict[str, str]. But got: {type(model_name)}")

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        """Returns the list of model names that should be included from server in cross site validation.add().

        Args:
            fl_ctx (FLContext): FL Context object.

        Returns:
            List[str]: List of model names.
        """
        return list(self.model_file_name.keys())

    def locate_model(self, model_name, fl_ctx: FLContext) -> DXO:
        dxo = None
        engine = fl_ctx.get_engine()

        if model_name in list(self.model_file_name.keys()):
            try:
                #Get model path and metadata
                job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
                run_dir = engine.get_workspace().get_run_dir(job_id)
                model_path = os.path.join(run_dir, self.model_dir)

                #model_load_path = os.path.join(model_path, self.model_file_name[model_name])
                model_load_path = "/home/ubuntu/joel/workspace/example_project/prod_00/admin@nvidia.com/transfer/cc85e6f1-abb4-461f-80a5-967c9f35e199/workspace/app_server/xgboost_model.json"
                xgmodel = None
                #self.log_info(fl_ctx, f"Load path: {model_load_path} (Check 0)")
            
                try:
                    #Load the xgboost json file from path into byte
                    with open(model_load_path, "r") as json_file:
                        xgmodel = json.load(json_file)
                        serialized_model = bytearray(json.dumps(xgmodel), "utf-8")
                    self.log_info(fl_ctx, f"Loaded {model_name} model from {model_load_path}. (Check 3)")
                except Exception as e:
                    self.log_error(fl_ctx, f"Unable to load XGB Model (Check 2): {secure_format_exception(e)}.")
                
                #Convert to dxo
                if xgmodel is not None:
                    weights = {"model_data": serialized_model}
                    dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights, meta={})
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load XGB Model (Check 1): {secure_format_exception(e)}.")
        self.log_info(fl_ctx, "Log 1133")
        
        return dxo
