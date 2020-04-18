# predictive-maintenance-pytorch
Deep Learning applied to condition based monitoring of a complex hydraulic system

## Backround
The data set was experimentally obtained with a hydraulic test rig. This test rig consists of a primary working and a secondary cooling-filtration circuit which are connected via the oil tank [1]. The system cyclically repeats constant load cycles (duration 60 seconds) and measures process values such as pressures, volume flows and temperatures while the condition of four hydraulic components (cooler, valve, pump and accumulator) is quantitatively varied. 
<p align="center">
  <img width="460" src="images/Schematic.png">
</p>

## Dataset
The data set is the [Condition monitoring of hydraulic systems Data Set](http://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems) contained in UCI Machine Learning Repository. It addresses the condition assessment of a hydraulic test rig based on multi sensor data shown in the schematic above. Four fault types are superimposed with several severity grades impeding selective quantification.

## Data Preparation
Data samples consist of a sequence of data acquired by different sensors during several cycles with duration of 60 seconds. Different sensors have different sampling rates of 1, 10, and 100 Hz which translate into sequence lenghts of respectively 60, 600, and 6000 samples. All the sequences are interpolated to a single sampling rate in [data_preparation](https://github.com/France1/predictive-maintenance-pytorch/blob/master/data_preparation/prepare_data.py) to feed different sensors to the same model. Downsampling is taken into account as it needed in LSTM models which starts to perform poorly for long sequences. From the original data set, sensor whose correlation is above 95% are eliminated as shown in this [notebook](https://github.com/France1/predictive-maintenance-pytorch/blob/master/notebooks/Data_preparation.ipynb). This reduced the sensors to a significant set 
```
{'CP','FS1','PS1','PS2','PS3','PS4','PS5','SE', 'VS1','profile'}
```
where `profile` contains the severity grade classes for each fault type.

## Multi-task Condition Based Classification
The problem is approached as a multi-task classification where each task is represented by the type of fault which contain the severity grade to classify through a multi-task loss function. Two different backbones are used to extract the features
### 1D CNN
The 1D CNN architecture is taken from [2] consisting of two convolutional and max pool layer. Batch normalization has been added to facilitate training
### LSTM with attention
For comparison an LSTM architecture has been tried. A vanilla LSTM architecture was able to achive good accuracy in a single-task problem, but failed to classify the 2nd failure mode in multi-task mode under all the hyperparameter combinations. Good performances where achived only after adding the attention mechanism in [3] with pytorch implementation inspired from this [project](https://github.com/prakashpandey9/Text-Classification-Pytorch)

## Installation
Pytorch models are trained using the nvcr.io/nvidia/pytorch:19.03-py3 Docker image from the [NVIDIA repository](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)

Run the container from the project folder
```
sudo nvidia-docker run -it --name pdm-pytorch --ipc=host \ 
  -v "$(pwd)":/home/project \
  -p <jupyter-port>:8888 \
  -p <tensorboard-port>:6006 \
  nvcr.io/nvidia/pytorch:19.03-py3
```
where `<jupyter-port>` and `<tensorboard-port>` are the port where you want to access to the jupyter notebook and tensorboard respectively.

Access the container shell, install and run jupyterlab
```
docker exec -it pdm-pytorch /bin/bash
pip install jupyterlab
jupyter-lab --ip 0.0.0.0 --no-browser --allow-root
```

Innstall and run tensorboard
```
pip install tensorboardX
tensorboard --logdir runs --bind_all
```



## References 
[1] N. Helwig, et al., "Condition monitoring of a complex hydraulic system using multivariate statistics," 2015 IEEE International Instrumentation and Measurement Technology Conference (I2MTC) Proceedings, Pisa, 2015, pp. 210-215.

[2] Ye Yuan, et al., A general end-to-end diagnosis framework for manufacturing systems, National Science Review, Volume 7, Issue 2, February 2020, Pages 418–429, https://doi.org/10.1093/nsr/nwz190 

[3] M.Luong et al, Effective Approaches to Attention-based Neural Machine Translation, 2015, arXiv:1508.04025


