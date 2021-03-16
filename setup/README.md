
In order to use this code, you need to use the docker image we have relased along with the code for dependencies. For a basic tutorial on Docker, see [this](https://www.tutorialspoint.com/docker/docker_overview.htm). For installation steps, follow [this](https://www.tutorialspoint.com/docker/installing_docker_on_linux.htm). In simple terms, for our case, docker is like a self-enclosed virtual environment which anyone can download and run the code inside it instead of having to follow the tedious steps for installing dependencies. Check the installation by running:

```bash
docker --version
```


### Pre-requisites

**Creating an account on Weights and Biases**: We use W&B (or wandb) for logging and visualization. If you do not have a [W&B](https://app.wandb.ai/) account, you can create one for free. After that, copy your W&B API key from Settings. This will be needed later on.

**Create a dockerhub account and login**: Create an account on [dockerhub.com](https://hub.docker.com/). In your local terminal, log in:
```bash
docker login -u <your-dockerhub-username>
Password: <Enter your dockerhub password when asked to enter
```

**Setup SSH keys on GitHub**: Instead of entering password everytime you push code to GitHub, we recommend using SSH keys. Follow the steps here:
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
