#
# usage: nvflare provision [-h] [-p PROJECT_FILE] [-w WORKSPACE] [-c CUSTOM_FOLDER] [--add_user ADD_USER] [--add_client ADD_CLIENT]
#
# optional arguments:
# -h, --help                                               show this help message and exit
# -p PROJECT_FILE, --project_file PROJECT_FILE                 file to describe FL project
# -w WORKSPACE, --workspace WORKSPACE                          directory used by provision
# -c CUSTOM_FOLDER, --custom_folder CUSTOM_FOLDER    additional folder to load python code
# --add_user ADD_USER                                             yaml file for added user
# --add_client ADD_CLIENT                                       yaml file for added client
#
# 1) To create project.yml, run 'nvflare provision' and choose either option 1 or 2
# 2) To create workspace, run 'nvflare provision -p project.yml'
#
nvflare provision -p project.yml