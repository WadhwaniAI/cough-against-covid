# usage: bash jupyter.sh 8001
# starts a jupyter lab session on port 8001 (8001 should be open on docker, if running on docker)

tmux new -d -s JupLabSession
tmux send-keys -t JupLabSession.0 "cd ../; jupyter lab --no-browser --ip 0.0.0.0 --port $1 --NotebookApp.password='sha1:2886cd70fbe8:f9e30a74d8464ac7320cbd6f81566093ac8e1cb2'" ENTER
