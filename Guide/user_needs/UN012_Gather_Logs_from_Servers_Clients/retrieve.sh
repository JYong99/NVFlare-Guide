directory_name="All_Logs"
job_id="f3a0227a-d013-4724-b681-df5efdfc7859"

serverpath="/home/ubuntu/joel/workspace/DxD_FL/prod_00/admin@dxd.com/transfer/$job_id/workspace/log.txt"

client1IP="ec2-54-179-196-255.ap-southeast-1.compute.amazonaws.com"
client1name="DxD_site_1"
client1path="/home/ubuntu/joel/$client1name/log.txt"

client2IP="ec2-13-250-27-156.ap-southeast-1.compute.amazonaws.com"
client2name="DxD_site_2"
client2path="/home/ubuntu/joel/$client2name/log.txt"

# Check if the directory exists
if [ -d "$directory_name" ]; then
    echo "Directory '$directory_name' already exists."
else
    # Create the directory
    mkdir "$directory_name"
    echo "Directory '$directory_name' created."
fi

cp $serverpath /home/ubuntu/joel/$directory_name/server_log.txt

scp -i /home/ubuntu/joel/nvflare.pem -r $client1IP:$client1path /home/ubuntu/joel/All_Logs/${client1name}_log.txt
scp -i /home/ubuntu/joel/nvflare.pem -r $client2IP:$client2path /home/ubuntu/joel/All_Logs/${client2name}_log.txt
