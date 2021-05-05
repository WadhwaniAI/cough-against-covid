## Instructions for setup

In order to use this code, you need to use the docker image we have relased along with the code for dependencies. For a basic tutorial on Docker, see [this](https://www.tutorialspoint.com/docker/docker_overview.htm). For installation steps, follow [this](https://www.tutorialspoint.com/docker/installing_docker_on_linux.htm). In simple terms, for our case, docker is like a self-enclosed virtual environment which anyone can download and run the code inside it instead of having to follow the tedious steps for installing dependencies. Check the installation by running:

```bash
$ docker --version
Docker version 19.03.12, build 48a66213fe
```

Depending on whether you have a CPU-only machine or a GPU machine, you need to use different docker images. If you do have a GPU machine, make sure you are able to run `nvidia-smi` and see your GPUs. 


### Pre-requisites

1. **Creating an account on Weights and Biases**: We use W&B (or wandb) for logging and visualization. If you do not have a [W&B](https://app.wandb.ai/) account, you can create one for free. After that, copy your W&B API key from Settings. This will be needed later on.

2. **Create a dockerhub account and login**: Create an account on [dockerhub.com](https://hub.docker.com/). In your local terminal, log in:
```bash
docker login -u <your-dockerhub-username>
Password: <Enter your dockerhub password when asked to enter>
```

<!-- 
3. **Create an account on Kaggle and store API key**: In order to download some of the public datasets used in this project (e.g. FreeSound Dataset), you need to setup Kaggle account and store your API key. Please follow instructions given [here](https://github.com/Kaggle/kaggle-api). Your API key file must reside at `~/.kaggle/kaggle.json`.
 -->

3. **Setup SSH keys on GitHub**: Instead of entering password everytime you push code to GitHub, this code requires using SSH keys. If you have already set this up, you can skip this step. Follow the steps [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

<!-- 
* Start an SSH agent: 
  ```bash
  eval `ssh-agent`
  ```
* Check if it has identities 
  ```bash
  ssh-add -l
  ```
* Generate a key pair with your email ID
  ```bash
  ssh-keygen -t rsa -b 4096 -C "piyush@wadhwaniai.org"
  ```
* Copy the contents of the file: `cat ~/.ssh/id_rsa.pub`
* Go to `github.com > Settings> Add SSH and GPG keys > Add new key`. Add a name to the key and paste the content and save it.
* Setup your credentials

  ```bash
  git config --global user.name "Piyush"
  git config --global user.email "piyush@wadhwaniai.org"
  ```
-->

4. **Setup data and output folders**: Create the following folder structure (to store data and outputs) inside `~/` or any other directory.

```bash
cac/
├── data
└── outputs
    └── <your name>

3 directories, 0 files
```


### Get started

* **Clone the repository**
```bash
cd ~/
mkdir projects; cd projects;
git clone git@github.com:WadhwaniAI/cough-against-covid.git
```
> For internal testing, please switch to branch `pb/test-no-lfs` to continue.

* **Pull required docker image**: Note that this might change depending on whether this image works. For a CPU-only machine, run:
```bash
docker pull wadhwaniai/cough-against-covid:py3-1.1
```
For a GPU machine, run:

> Note: This image is not ready yet.

```bash
docker pull wadhwaniai/cough-against-covid:py3-1.2
```

* **Fire a container using the docker image**: First, set the following enviroment variables:
```bash
export WANDB_API_KEY=<your W&B API key obtained from previous section>
export WANDB_CONFIG_DIR=</path/to/any/folder>/.config/wandb/
```
> Tip: It is convenient to put these in your `~/.bashrc` or `~/.bash_profile` instead of setting them manually everytime.
> Add the line `export WANDB_API_KEY=XYZ` to your `~/.bashrc` file.

Depending on whether you have a CPU-only machine or a GPU machine, you can choose which docker image to use.
Next, you can start a container by the following command: 
```bash
cd ~/projects/cough-against-covid/setup/

# for a GPU machine
bash create_container.sh -g 0 -n sample-container -e ~/cac/ -u piyush -p 8001

# for a CPU machine
bash create_container.sh -g -1 -n sample-container -e ~/cac/ -u piyush -p 8001


>>> Explanation
-g: GPU number, pass -1 for a non-GPU machine
-n: name of the container
-e: path to the folder where data and outputs are to be stored
-u: username (this is the name of folder you created inside outputs/ folder)
-p: port number (this is needed if you want to start jupyter lab on a remote machine)
```

Once you are inside the container, you can run the training/evaluation scripts. You should see an interface like this.
![image](https://user-images.githubusercontent.com/51699359/111505444-a0de2f00-876e-11eb-9cf1-67b070446ab8.png)

Note that, inside the container, the code is mounted at `/workspace/cough-against-covid/`, the data is mounted at `/data/` and your outputs at `/outputs/`.

For CPU-only machine, inside the container, run the following command to test you are able to load the right `torch` version:
```bash
$ python -c "import torch; print(torch.__version__)"
1.6.0+cpu
```

For GPU machine, inside the container, run the following command to test you are able to load the right `torch` version:
```bash
$ python -c "import torch; print(torch.__version__)"
1.6.0
```

You can exit (and kill) the running docker container by pressing `Ctrl + D` inside the container.

#### (Optional) Starting Jupyter lab
In order to spin up jupyter lab from inside the container, use: (note the use of the same port which was used to start the container)
```bash
cd /workspace/cough-against-covid/setup/
bash jupyter.sh 8001
```
Now visit `<IP of the machine>:8001` in a browser. If running on a local machine, visit `0.0.0.0:8001`, the password is "cac@1234" without the double-quotes. If you want to keep this secure, you can set the password by running `jupyter notebook passwd`, modifying `jupyter.sh` to remove `--NotebookApp.password` argument before running the above command.


#### (Optional) Updating docker image
In case you want to make certain changes to your local docker container and commit them as a new image, you can do that by modifying the Dockerfile and then running:
```bash
cd setup/
docker build -t wadhwaniai/cough-against-covid:py3-1.0 . -f Dockerfile
docker push <your docker username>/<your docker repo>:<tag>
```
