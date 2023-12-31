Official code for ICCV 2023 paper: [GAIT: Generating Aesthetic Indoor Tours with Deep Reinforcement Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_GAIT_Generating_Aesthetic_Indoor_Tours_with_Deep_Reinforcement_Learning_ICCV_2023_paper.pdf)

## Installing Dependencies
Since we install `habitat-sim` by building from source, the `environment.yml` file we provide could not be directly used but can serve as reference.
1. Start by creating an conda environment with python 3.7, as required by `habitat-sim`: `$ conda create -n habitat python=3.7 cmake=3.14.0`
2. Then, install pytorch. If `habitat-sim` is installed before pytorch, then pytorch would be installed as cpu-only. `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
3. clone [habitat-sim](https://github.com/facebookresearch/habitat-sim) github. 
4. Edit the `habitat-sim` source code. By default, habitat-sim has "gravity" enabled, so if we move the agent upwards, it falls down to the ground by itself. We change its source code to disable this.
   1. `$ cd habitat-sim/src_python/habitat_sim`.
   2. Edit `simulator.py`. Search for `step(`, there are three overloaded methods called `step()`. For each one, add a parameter `apply_filter=True`. In the third `step()` definition, go to the line calling `agent.act()`, and append `apply_filter=apply_filter` as a parameter.
   3. Edit `agent/agent.py`. Search for `act(`, append `apply_filter=True` as a parameter. In the `act()` method, there are two calls of `self.controls.action()`. Append `apply_filter=apply_filter` to each of them.
4. Install `habitat-sim` from source. `habitat-sim` is available as a conda pacakge, but we have met some issues that could be solved by installing from source. Follow this [guide](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source) from `habitat-sim` Github. 
   1. If you have `nvcc`: `$ export CUDACXX=/usr/local/cuda/bin/nvcc`. Otherwise, install `nvcc` via conda and then export path to it: `$sudo apt install libxml2`, `$ conda install -c conda-forge cudatoolkit-dev`, `$ export CUDACXX=/home/username/anaconda3/envs/habitat/pkgs/cuda-toolkit/bin/nvcc`
   2. `sudo apt install build-essential`
   3. For `$ python setup.py install`: use flag `--with-cuda`, and include `--headless` flag as needed.
5. While you are building `habitat-sim` from source, take the time to download the `Replica` Dataset
   1. `$ git clone https://github.com/facebookresearch/Replica-Dataset.git`
   2. `$ sudo apt install pigz`
   3. Make sure `Replica-Dataset` and `gait` are in the same folder. Then `$cd Replica-Dataset`, `$./download.sh ../gait/habitat_data/Replica`
   4. `$ wget http://dl.fbaipublicfiles.com/habitat/sorted_faces.zip`, `$ unzip sorted_faces.zip`
   5. `$./sorted_faces/copy_to_folders ~/aestheticview/habitat_data/Replica/`
6. Install `conda` dependencies in `environment.yml`
7. Install `pip` dependencies
   1. `$ pip install gym tensorboard dm-env termcolor cma`
   2. `$ pip install -U ray`
   3. `$ pip install hydra-core --upgrade`
   4. `$ pip install hydra-submitit-launcher --upgrade`
8. Download the pretrained weights of the aesthetics model (View Evaluation Net) from https://github.com/zijunwei/ViewEvaluationNet, and put `EvaluationNet.pth.tar` in `gait/snapshots/params`

## Training
All hyperparameters are set to the default condition.
`cfgs/config.yaml` contains the hyperparameters for GAIT environment and for DrQv2.
Specifically, `ray` enables or disables multi-GPU training, `diversity, smoothness, constant_noise, no_aug` enables/disables the corresponding ablation setting as their name suggests.
`cfgs/task/single_train.yaml` specifies the algorithm (DrQv2 or CURL) and the scene.
`cfgs/task/medium.yaml` specifies the Data Steps and Linearly decayed noise schedule for DrQv2)
`cfgs/task/cma-es.yaml` specifies cma-es specific hyperparameters.

To train GAIT-DrQ-v2
`$ bash drqtrain.sh`

To train GAIT-CURL
`$ bash curltrain.sh`

To run CMA-ES
`$ bash cmatrain.sh`

Use the corresponding watch script to watch the training output, e.g. `$ bash drqwatch.sh`

## Evaluation
We provide 1 pre-trained checkpoint corresponding to `DrQ-v2 default room0` located in `logs/drqv2_habitat/66 16  multi 1m decay 3m train room_0_`, containing only the actor network, and the training configurations.

To generate figures, sequence frames, and the corresponding interpolated videos in the paper with the pre-trained weights, first edit `final_evaluation.py` to choose which figure, then 

`$ python final_evaluation.py`

Note that for training and evaluation, `hydra` will change the working directory to `gait/logs/${algo}/${now:%Y.%m.%d}_${now:%H%M%S}`.

## Files
+ `rlcam_drqv2_mql.py` and `curl_train.py` contains the main training code.
+ `drqv2_net.py` contains the pytorch network architecture definitions for the Actor and the Critic networks.
+ `habitat_test.py` contains our wrappers based on `habitat-sim`, which handles camera pose, computes reward, etc (Sections 3.1 and 3.2 of the paper.)
+ `drqv2/` and `curl/` contains the original code from DrQ-v2 and CURL respectively. Our network architectures defined in `drqv2_net.py` are based on `drqv2/drqv2.py`.