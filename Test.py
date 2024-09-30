import json
from nvflare.apis.dxo import DXO, DataKind, from_shareable

model_load_path = "/home/ubuntu/joel/workspace/example_project/prod_00/admin@nvidia.com/transfer/cc85e6f1-abb4-461f-80a5-967c9f35e199/workspace/app_server/xgboost_model.json"
xgmodel = None

with open(model_load_path, "r") as json_file:
    xgmodel = json.load(json_file)
    serialized_model = bytearray(json.dumps(xgmodel), "utf-8")


weights = {"model_data": serialized_model}
dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights, meta={})
transfer = DXO.to_shareable(dxo)
print(transfer)
