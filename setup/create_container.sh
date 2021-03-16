# example command to run
# bash create_container.sh -g 0 -n sample-container -e /Users/piyushbagad/cac/ -u piyush -p 8001
# -g: GPU number
# -n: name of the container
# -e: path to the folder where data and outputs are to be stored
# -u: username (this is the name of folder you created inside outputs/ folder)
# -p: port number (this is needed if you want to start jupyter lab on a remote machine)


# set docker image
image=wadhwaniai/covid:cac-aws-1.0-py3

# get inputs
while getopts "g:n:u:e:p:" OPTION; do
    case $OPTION in
        g) gpu=$OPTARG;;
		n) name=$OPTARG;;
		u) user=$OPTARG;;
		e) efs_path=$OPTARG;;
        p) port=$OPTARG;;
        *) exit 1 ;;
    esac
done

# check if ENV variables are set
if [[ -z $WANDB_API_KEY ]] ; then
	echo "ERROR: set the environment variable WANDB_API_KEY"
	exit 0
fi

if [[ -z $WANDB_CONFIG_DIR ]] ; then
	echo "ERROR: set the environment variable WANDB_CONFIG_DIR"
	exit 0
fi

# start the docker container
NV_GPU=$gpu nvidia-docker run --rm -it \
	--name "$gpu"_"$name" \
    -p $port:$port \
	-v $PWD/../:/workspace/cac-test-release \
	-v $efs_path/outputs/$user:/output \
	-v $efs_path/outputs:/all-output \
	-v $efs_path/data:/data \
	--env WANDB_DOCKER=$image \
	--env WANDB_API_KEY=$WANDB_API_KEY \
	--ipc host \
	$image bash
