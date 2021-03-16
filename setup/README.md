
In order to use this code, you need to use the docker image we have relased along with the code for dependencies. For a basic tutorial on Docker, see [this](https://www.tutorialspoint.com/docker/docker_overview.htm). For installation steps, follow [this](https://www.tutorialspoint.com/docker/installing_docker_on_linux.htm). In simple terms, for our case, docker is like a self-enclosed virtual environment which anyone can download and run the code inside it instead of having to follow the tedious steps for installing dependencies. Check the installation by running:

```bash
docker --version
```
This code also assumes having a GPU machine. Make sure you are able to run `nvidia-smi` and see your GPUs.


### Pre-requisites

**Creating an account on Weights and Biases**: We use W&B (or wandb) for logging and visualization. If you do not have a [W&B](https://app.wandb.ai/) account, you can create one for free. After that, copy your W&B API key from Settings. This will be needed later on.

**Create a dockerhub account and login**: Create an account on [dockerhub.com](https://hub.docker.com/). In your local terminal, log in:
```bash
docker login -u <your-dockerhub-username>
Password: <Enter your dockerhub password when asked to enter
```

**Setup SSH keys on GitHub**: Instead of entering password everytime you push code to GitHub, we recommend using SSH keys. If you have already set this up, you can skip this step. Follow the steps here:
* Start an SSH agent: 
  ```bash
  eval `ssh-agent`
  ```
* Check if it has identities 
  ```bash
  ssh-add -l
  ```
* Generate a key pair with your Wadhwani AI email ID
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
* **Setup data and output folders**: This section assumes that you have followed steps here to download and store dataset(s) in a specific structure. For example, suppose you have a common folder at `/Users/piyushbagad/cac/`. The data and outputs will reside at `/Users/piyushbagad/cac/data/` and `/Users/piyushbagad/cac/outputs/` respectively. Next, in the `outputs/` folder, create a folder by your name (e.g. `piyush/`).


### Get started

* **Clone the repository**
```bash
cd ~/
mkdir projects; cd projects;
git clone git@github.com:WadhwaniAI/cac-test-release.git
```

* **Pull required docker image**: Note that this might change depending on whether this image works.
```bash
docker pull wadhwaniai/covid:cac-aws-1.0-py3
```

* **Fire a container using the docker image**: First, set the following enviroment variables:
```bash
export WANDB_API_KEY=<your W&B API key obtained from previous section>
export WANDB_CONFIG_DIR=</path/to/any/folder>/.config/wandb/
```
Next, you can start a container by the following command: 
```bash
cd ~/projects/cac-test-release/setup/
bash create_container.sh -g 0 -n sample-container -e /Users/piyushbagad/cac/ -u piyush -p 8001

>>> Explanation
-g: GPU number
-n: name of the container
-e: path to the folder where data and outputs are to be stored
-u: username (this is the name of folder you created inside outputs/ folder)
-p: port number (this is needed if you want to start jupyter lab on a remote machine)
```

Once you are inside the container, you can run the training/evaluation scripts. Note that, inside the container, the code is mounted at `/workspace/cac-test-release/`, the data is mounted at `/data/` and your outputs at `/outputs/`.

(Optional) In order to spin up jupyter lab from inside the container, use: (note the use of the same port which was used to start the container)
```bash
cd /workspace/cac-test-release/setup/
bash jupyter.sh 8001
```