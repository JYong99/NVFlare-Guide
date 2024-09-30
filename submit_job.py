from nvflare.fuel.flare_api.flare_api import new_secure_session, basic_cb_with_print

sess = new_secure_session(
    username="admin@nvidia.com",
    startup_kit_location="/home/ubuntu/joel/workspace/example_project/prod_00/admin@nvidia.com",
)

try:
    path_to_example_job = "/home/ubuntu/joel/Jobs/Xgboost_CL1"
    job_id = sess.submit_job(path_to_example_job)
    print(job_id + " was submitted")
    sess.monitor_job(job_id, cb=basic_cb_with_print, cb_run_counter={"count":0})
finally:
    sess.close()