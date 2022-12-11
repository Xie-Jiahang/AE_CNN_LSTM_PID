# AE_CNN_LSTM_PID

Battery thermal management is essential to achieve good performance and a long battery system lifespan in electric vehicles and stationary applications. Such a thermal management system is dependent on temperature monitoring, which is frequently hampered by the limited sensor measurements.
The virtual sensor is brought forward to overcome this physical restriction and provide broader access to the battery's temperature distribution.
Through leveraging the combined convolutional neural network (CNN) and long short-term memory (LSTM) networks to extract both spatial and temporal information from the data, this paper proposes a novel virtual sensing platform. A PID compensator is included to offer auxiliary correction to the inputs and drive the prediction error to zero over time in a feedback loop. Off-line and online modes of this CNN-LSTM virtual sensor are considered. The network, which is trained off-line, will work with the PID compensator in the online mode with real-time sensor data. With the PID-based accuracy-boosted virtual sensor, the performance of the trained CNN-LSTM prediction on real-time data inputs is improved. Besides, this PID compensator reduces the number of hyper-parameters to be tuned. Based on control theory, the design of PID and its analysis are presented as well.

# Key words
Virtual sensor, PID, Convolutional Neural Network - Long Short-Term Memory Networks, Battery energy storage system, Tree-structured parzen estimator, Compressed sensing

# Code and simulation description
Input: battery surface temperature dataset

main-ml.py: training of CNN-LSTM, as the proposed virtual sensor

main-control.py: PID based accuracy boosing algorithm of trained CNN-LSTM

pid.py: implement the PID compensator algorithm under this use case

# Citation
If the dataset, simulation or code is used in your paper/experiments, please cite the following paper.
```
@article{XIE2023120424,
title = {PID-based CNN-LSTM for accuracy-boosted virtual sensor in battery thermal management system},
journal = {Applied Energy},
volume = {331},
pages = {120424},
year = {2023},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2022.120424},
url = {https://www.sciencedirect.com/science/article/pii/S0306261922016816},
author = {Jiahang Xie and Rufan Yang and Hoay Beng Gooi and Hung Dinh Nguyen},
}
```
