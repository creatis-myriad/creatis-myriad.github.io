---
layout: post
title:  "Jean Zay Ultimate Guide"
author: 'Hang Jung Ling'
date:   2022-05-31
categories: JeanZay, Guide
---

- [**Introduction**](#introduction)
- [**Jean Zay account application**](#jean-zay-account-application)
  - [Signup on the eDARI portal <a name="idris-account"></a>](#signup-on-the-edari-portal-)
  - [Project creation](#project-creation)
- [**First connection on Jean Zay**](#first-connection-on-jean-zay)
- [**Quick look at the storage spaces on Jean Zay**](#quick-look-at-the-storage-spaces-on-jean-zay)
- [**Conda environment on Jean Zay** <a name="conda-install"></a>](#conda-environment-on-jean-zay-)
- [**Launch JupyterLab/Notebook on Jean Zay**](#launch-jupyterlabnotebook-on-jean-zay)
- [**Job submission**](#job-submission)
  - [Interactive job](#interactive-job)
  - [Batch job](#batch-job)
- [**TODO**](#todo)

&nbsp;

## **Introduction**
Jean Zay is a supercomputer converged platform acquired by the French Ministry of Higher Education, Research and Innovation through the intermediary of the French civil company, GENCI (Grand Equipement National De Calcul Intensif). More history of Jean Zay platform can be found [here](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html). 

Jean Zay offers the possibility to extend the classic usage modes of high performance computing to new usages in artificial intelligence (AI). Jean Zay might be a great alternative to Creatis's own cluster if you intend to run multiple codes at once to prevent the overload of the cluster. All CPU, GPU, and software configurations of Jean Zay are detailed [here](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-hw-eng.html).

&nbsp;

## **Jean Zay account application**
Usually, all members of Creatis are eligible to apply their own Jean Zay's account. Before getting too excited about the application, it is important to know that there are two types of access:
1. Dynamic Access (AD):  
   * If your desired number of hours for CPU or GPU ≤ 50,000  
   * Requests for resources may be made throughout the year and are renewable 
2. Regular Access (AR):  
   * If your desired number of hours for CPU or GPU > 50,000
   * Application is only open at a specific time

More information about Dynamic Access and Regular Access can be found [here](http://www.idris.fr/eng/info/gestion/demandes-heures-eng.html#to_develop_algorithms_in_artificial_intelligence_ai_on_the_jean_zay_gpu_partition). Only procedures to apply Dynamic Access for GPU resources will be detailed in this guide. For other purposes, please contact [Thomas Baudier](mailto:Thomas.Baudier@creatis.insa-lyon.fr).


### Signup on the eDARI portal <a name="idris-account"></a>
The first thing to do is creating an account on [eDARI](https://www.edari.fr/). The next step to create an IDRIS account after logging in on [eDARI](https://www.edari.fr/).  

![](/collections/images/Jean_Zay_Guide/IDRIS.jpg)

It is quite simple to fill the application form. Most of the time, the details of Creatis lab are pre-filled when you select Creatis for the Research Lab section. Some important information is included below just in case:
1. Directeur du laboratoire  
   * M. BEUF Olivier
   * olivier.beuf@creatis.insa-lyon.fr
2. Responsable sécurité de l'utilisateur
   * M. BELLET Fabrice
   * fabrice.bellet@creatis.insa-lyon.fr
   * 0472436142
3. Adresse IP (Internet Protocol) de connexion & FQDN (Fully Qualified Domain Name)
   * 195.220.108.5 (If you have a laptop) & tux.creatis.insa-lyon.fr
   * Use [this](http://monip.org/) to get your IP address and FQDN if you have a desktop PC
   * It is recommended to use tux's IP address even if you own a desktop PC so that you can access to Jean Zay everywhere via tux

You will also have to fill the first 8 characters of your [eDARI](https://www.edari.fr/)'s password.

> 📝 These 8 characters will be the last 8 characters of your first Jean Zay's password. 

> ⚠️ Don't forget to validate your information by clicking on click on "Valider la saisie des informations"! 
> ![](/collections/images/Jean_Zay_Guide/computing-account-validate.jpg)

After the submission of the application form, you will soon receive an email to fill another online questionnaire and upload your CV.


### Project creation
While waiting for the confirmation of your IDRIS account, you can start creating your projects by clicking on `Créer ou renouveler un dossier`. If you wish to collaborate on a project, you can attach to a project created by one of your collaborators by clicking on `Se rattacher à un dossier` and then entering the project number. 

![](/collections/images/Jean_Zay_Guide/project.jpg)

In the form you have to describe your project, the type of data you use, the resources you plan to request etc. A project proposal is needed if you request more than 10k GPU hours. Again, the *Correspondant technique* is Fabrice BELLET.     

> ⚠️ Don't forget to validate your information by clicking on click on "Valider la saisie des informations"!
> ![](/collections/images/Jean_Zay_Guide/project-validate.jpg)

Congratulations! You have finally completed all the administrative procedures to request your access to Jean Zay. All you have to do now is wait! 😄 Yup, wait for roughly 1-2 months before getting your username and password. 

> 📝 For non-French applicant, the delay might be slightly longer as CNRS has to review your file for security purpose (FSD, ASSAV...).

&nbsp;

## **First connection on Jean Zay**
If you are here, that means your application is successful, and you've received your Jean Zay username and password. 👏👏 Yes, the username is weird. Don't be surprised if you get something like `ubz52ne`, it is completely normal.

Jean Zay platform is only accessible via the IP address that you gave during the [creation of your IDRIS account](idris-account). To do so, simply type the following command in a terminal or Putty for Windows users:

```shell
# For those who filled in tux's IP address. Skip this if you've filled in your PC's IP address
ssh <username>@tux.creatis.insa-lyon.fr # Replace <username> by your Creatis username, e.g. abc@tux.creatis.insa-lyon.fr

# Connect to Jean Zay platform
ssh <jean-zay-username>@jean-zay.idris.fr # Replace <jean-zay-username> by your Jean Zay username, e.g. ubz52ne@jean-zay.idris.fr
```

After typing those commands, you will be asked to type your password. Remember the password that you've received by email and the 8 characters that you've filled in during the [creation of your IDRIS account](idris-account)? Yes, your login password is the combination of these two. 😂 Don't be worried, you will get to change your password right after this. 

Once you've typed the first login password and changed the password, you should see something like this:

```
Last login: Wed Jun  1 06:34:27 2022 from 195.220.108.5
***********************************************************************
* Ceci est un serveur de calcul de l'IDRIS. Tout acces au systeme     *
* doit etre specifiquement autorise par l'IDRIS. Si vous tentez de    *
* de continuer a acceder cette machine alors que vous n'y etes pas    *
* autorise, vous vous exposez a des poursuites judiciaires.           *
*                               ---                                   *
* This is an IDRIS compute node.  Each access to this system must be  *
* properly authorized by IDRIS. If you go on accessing this machine   *
* without authorization, then you are liable to prosecution.          *
***********************************************************************
*                                                                     *
* Orsay      CNRS / IDRIS - Frontale  -  jean-zay.idris.fr   France   *
*                                                                     *
***********************************************************************
```

You've now successfully connect to Jean Zay platform. Don't get too excited because you've to wait for another 12-24 hours before running any code as Jean Zay has to verify your IP address. 😄

For future connections, you can continue to use your password. However, I would recommend you to log in using your private ssh key. To do so, you've to store your public key in the `authorized_keys` file on Jean Zay and your private key in `~/.ssh/` on tux or your local machine. 

To transfer your public key into the `authorized_keys` file on Jean Zay, simply type these on your local machine: 

```shell
# Make sure your private key has the correct permission
chmod 600 ~/.ssh/id_rsa

# Transfer your public key (~/.ssh/id_rsa.pub) into the authorized_keys file on Jean Zay
ssh-copy-id <jean-zay-username>@jean-zay.idris.fr
```

To type your passphrase only one time using the ssh-agent program, enter the two following commands on your local machine or tux:

```shell
eval 'ssh-agent'
ssh-add ~/.ssh/id_rsa
```

> ⚠️ Connection without entering the passphrase is limited to the lifespan of the agent. You will need to re-launch the agent at each new session.
 
More detailed guide can be found [here](http://www.idris.fr/eng/faqs/ssh_keys-eng.html). From now on, you can connect to Jean Zay without entering your password. 😉

&nbsp;

## **Quick look at the storage spaces on Jean Zay**
Jean Zay offers several spaces to store your dataset or repository. I will only cover 3 spaces here. For more details, pleaser refer [here](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-calculateurs-disques-eng.html).
1. `$HOME or /linkhome/rech/gencre01/<jean-zay-username>/`
   * Only 3Gb
   * Use to store your config files or repository
2. `$WORK or /gpfswork/rech/<your-project-account>/<jean-zay-username>/`
   * Bigger space: 5Tb.
   * 100Gb/s read/write speed
   * Limited in inodes (500k). You will reach this quota very soon if you have a lot of folders/subfolders.
   * `<your-project-account>` is usually consisted of 3 alphabets, e.g. `abc`. Pay extra attention if you have multiple projects. Make sure you work in the correct project space.
     * It is possible to know your project account through [extranet](https://extranet.idris.fr). To do so, type `passextranet` to create your [extranet](https://extranet.idris.fr) password and wait up to 30 minutes before you can log in at [extranet](https://extranet.idris.fr).  
   * If you have any collaborator, the shared space in `$ALL_CCFRWORK or /gpfswork/rech/<your-project-account>/commun/` might be useful for data sharing. 
3. `$SCRATCH or /gpfsscratch/rech/<your-project-account>/<jean-zay-username>/`
   * Very large space (2.5Pb) shared by all users
   * 500Gb/s read/write speed
   * Will be deleted after 30 days of inactivity 

`idrquota` can be used to track your spaces usage.
```
usage: idrquota [-h] [-t {Kio,Mio,Gio,Tio,Pio}] [-m] [-u USERNAME] [-w] [-s] [-p PROJECT]

Display quota information for the user or the projects.

Normal users can display the quotas of their home directory and the work/store
quotas for their projects.

Privileged users can display the quotas of other users and projects.

optional arguments:
  -h, --help            show this help message and exit
  -t {Kio,Mio,Gio,Tio,Pio}, --unit {Kio,Mio,Gio,Tio,Pio}
                        the quota block unit (default: Gio)
  -m, --home            show the home quota of the user
  -u USERNAME, --username USERNAME
                        the username (default: current user)
                        only usable by privileged users
  -w, --work            show the work quota of the project
  -s, --store           show the store quota of the project
  -p PROJECT, --project PROJECT
                        the groupname of the project (default: current project)

```
&nbsp;

## **Conda environment on Jean Zay** <a name="conda-install"></a>
It is possible and easy to set up a conda environment on Jean Zay. A quick how to install `miniconda` in your `$WORK` directory:

```shell
# download Miniconda installer
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh
# install Miniconda
MINICONDA_PATH=$WORK/miniconda3
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
# make sure conda is up-to-date
source $MINICONDA_PATH/etc/profile.d/conda.sh
conda update --yes conda
# Update your .bashrc to initialise your conda base environment on each login
conda init
```

By default, the `.bashrc` is not executed when connecting on the cluster. To make sure it is run, add a file `~/.bash_profile` with
```shell
#
# ~/.bash_profile
#

[[ -f ~/.bashrc ]] && . ~/.bashrc
```

&nbsp;

## **Launch JupyterLab/Notebook on Jean Zay**
Yes, you are able to launch JupyterLab/Notebook on Jean Zay. It is not that straightforward and requires some tweaks. Some modules provided by Jean Zay already have JupyterLab/Notebook installed. Click [here](http://www.idris.fr/eng/jean-zay/pre-post/jean-zay-jupyter-notebook-eng.html) for more information.

In this section, I will only focus on how to correctly install Jupyter-related packages in your conda `base` environment. By doing this, you don't have to install JupyterLab/Notebook in every conda environment you create. Hang tight! 😃

1. Make sure you've installed `miniconda` in your `$WORK` directory, otherwise refer to [previous section](conda-install). 
2. Type `conda deactivate` to go to the `base` environment.
3. Install Jupyter Notebook (Optional if you only plan to use JupyterLab)
   ```shell
   conda install -c conda-forge notebook
   conda install -c conda-forge nb_conda_kernels
   ``` 
4. Install Jupyter Lab
      ```shell
   conda install -c conda-forge jupyterlab
   conda install -c conda-forge nb_conda_kernels # Ignore this if you've run this in Step 2
   ``` 
5. Install Jupyter extensions
   ```shell
   conda install -c conda-forge jupyter_contrib_nbextensions
   ```

Your `base` environment now supports JupyterLab/Notebook. To be able to switch environment after launching JupyterLab/Notebook in the `base` environment, make sure you install `ipykernel` in every future environment you create. A simple way to do so:

```shell
conda create -n <your-env-name> ipykernel
```

To launch JupyterLab/Notebook on Jean Zay, type:

```shell
# Launch JupyterLab
idrlab

# Or launch Jupyter Notebook
idrjup
```

After entering this, you will see something like this:
```shell
INFO 2019-11-22 12:10:08,916 Starting Jupyter server. Please wait ...:
INFO 2019-11-22 12:10:08,933 --Launching Jupyter server. Please wait before attempting to connect ...
INFO 2019-11-22 12:10:14,070 --Jupyter server launched. Please connect.
URL de connexion :     https://idrvprox.idris.fr
Mot de passe URL :     <mot de passe utilisateur>
Mot de passe jupyter : abc1defg2hijk3lmno4pqrs5tuvw6xyz
```

Next: 
* You have to open a browser from a machine for which you have declared the IP address and enter the displayed URL connection: `https://idrvprox.idris.fr`. 
* An Identification page will be loaded:  
     
  <p align="center">
   <img src="/collections/images/Jean_Zay_Guide/1-identification.jpg" />
  </p>
    
* Enter your Jean Zay identifiers (username and password) and click on `Login`.
* In the list of active sessions which appears, select the one you wish to reach by clicking on `Submit Query`.
       
  <p align="center">
   <img src="/collections/images/Jean_Zay_Guide/2-3-espace_censored.jpg" />
  </p>
    
* On the page which then displays, enter the random Jupyter password returned by the `idrjup` or `idrlab` command (in this example, `abc1defg2hijk3lmno4pqrs5tuvw6xyz`) and click on `Log in`. 
      
  <p align="center">
   <img src="/collections/images/Jean_Zay_Guide/4-idrpassword.jpg" />
  </p> 
    

Again, more detailed explanations can be found [here](http://www.idris.fr/eng/jean-zay/pre-post/jean-zay-jupyter-notebook-eng.html).

&nbsp;

## **Job submission**
For those who are familiar with Creatis cluster, the job submission on Jean Zay is quite similar. For job management on Jean Zay, `slurm` is used instead of `pbs`. You have the possibility to submit an interactive or a batch job.  

### Interactive job
The command to start an interactive bash terminal: 

```shell

srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread [--other-options] bash

```
More details can be found [here](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_interactif-eng.html).

### Batch job
To submit a batch job, you have to create a submission script `xxxx.slurm`. Here is an example for a job with 1 GPU in default GPU partition

```

#!/bin/bash
#SBATCH --job-name=single_gpu        # name of job
##SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition gpu_p2
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)
##SBATCH --cpus-per-task=3           # number of cores per task (with gpu_p2: 1/8 of the 8-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=00:10:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=gpu_single%j.out    # name of output file
#SBATCH --error=gpu_single%j.out     # name of error file (here, in common with the output file)
 
# cleans out the modules loaded in interactive and inherited by default 
module purge
 
# loading of modules
module load ...
 
# echo of launched commands
set -x
 
# code execution
./single_gpu_exe

```

&nbsp;

### **Feel free to [email me](mailto:hang-jung.ling@creatis.insa-lyon.fr) if you have any question! 😁** <!-- omit in toc -->

&nbsp;

## **TODO**
- [ ] File copy from Jean Zay to local machine and vice versa
- [ ] Finish Job submission section

