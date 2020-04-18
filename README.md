# predictive-maintenance-pytorch
Deep Learning applied to condition based monitoring of a complex hudraulic system

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

## References 
[1] N. Helwig, et al., "Condition monitoring of a complex hydraulic system using multivariate statistics," 2015 IEEE International Instrumentation and Measurement Technology Conference (I2MTC) Proceedings, Pisa, 2015, pp. 210-215.
