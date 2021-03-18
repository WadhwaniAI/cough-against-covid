## Instructions for setup

In order to use this code, you need to use the docker image we have relased along with the code for dependencies. For a basic tutorial on Docker, see [this](https://www.tutorialspoint.com/docker/docker_overview.htm). For installation steps, follow [this](https://www.tutorialspoint.com/docker/installing_docker_on_linux.htm). In simple terms, for our case, docker is like a self-enclosed virtual environment which anyone can download and run the code inside it instead of having to follow the tedious steps for installing dependencies. Check the installation by running:

```bash
docker --version
```
This code also assumes having a GPU machine. Make sure you are able to run `nvidia-smi` and see your GPUs. (This line maybe removed later when we have support for running on CPU machines)


### Pre-requisites

1. **Creating an account on Weights and Biases**: We use W&B (or wandb) for logging and visualization. If you do not have a [W&B](https://app.wandb.ai/) account, you can create one for free. After that, copy your W&B API key from Settings. This will be needed later on.

2. **Create a dockerhub account and login**: Create an account on [dockerhub.com](https://hub.docker.com/). In your local terminal, log in:
```bash
docker login -u <your-dockerhub-username>
Password: <Enter your dockerhub password when asked to enter>
```

3. **Setup SSH keys on GitHub**: Instead of entering password everytime you push code to GitHub, we recommend using SSH keys. If you have already set this up, you can skip this step. Follow the steps [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
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
  ``` -->

4. **Setup data and output folders**: In order to run code for this project, we expect a certain directory structure for storing dataset(s) and model outputs. For example, suppose you have a common folder at `/Users/piyushbagad/cac/`. The data and outputs will reside at `/Users/piyushbagad/cac/data/` and `/Users/piyushbagad/cac/outputs/` respectively. Next, in the `outputs/` folder, create a folder by your name (e.g. `piyush/`).
```bash
cac/
├── data
└── outputs
    └── piyush

3 directories, 0 files
```


### Get started

* **Clone the repository**
```bash
cd ~/
mkdir projects; cd projects;
git clone git@github.com:WadhwaniAI/cough-against-covid.git
```

> TEMPORARY: Please switch to pb/setup branch using `git checkout pb/setup` for further steps.

* **Pull required docker image**: Note that this might change depending on whether this image works.
```bash
docker pull wadhwaniai/cough-against-covid:py3-1.0
```

* **Fire a container using the docker image**: First, set the following enviroment variables:
```bash
export WANDB_API_KEY=<your W&B API key obtained from previous section>
export WANDB_CONFIG_DIR=</path/to/any/folder>/.config/wandb/
```
> Tip: It is convenient to put these in your ~/.bashrc or ~/.bash_profile instead of setting them manually everytime.

Next, you can start a container by the following command: 
```bash
cd ~/projects/cough-against-covid/setup/
bash create_container.sh -g 0 -n sample-container -e /Users/piyushbagad/cac/ -u piyush -p 8001

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

(Optional) In order to spin up jupyter lab from inside the container, use: (note the use of the same port which was used to start the container)
```bash
cd /workspace/cough-against-covid/setup/
bash jupyter.sh 8001
```
Now visit `<IP of the machine>:8001` in a browser. If running on a local machine, visit `0.0.0.0:8001`, the password is "cac@1234" without the double-quotes. If you want to keep this secure, you can set the password by running `jupyter notebook passwd`, modifying `jupyter.sh` to remove `--NotebookApp.password` argument before running the above command.


#### Updating docker image (Optional)
In case you want to make certain changes to your local docker container and commit them as a new image, you can do that by modifying the Dockerfile and then running:
```bash
cd setup/
docker build -t wadhwaniai/cough-against-covid:py3-1.0 . -f Dockerfile
docker push <your docker username>/<your docker repo>:<tag>
```
