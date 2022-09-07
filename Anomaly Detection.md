###### tags: `Azure` `Machine Learning` `python` 

Anomaly detection
===


## Resource
[PyOD open source](https://towardsdatascience.com/anomaly-detection-with-pyod-b523fc47db9)
[PyOD paper](https://www.jmlr.org/papers/volume20/19-011/19-011.pdf)


## Methodology
:::success
- [ADBench: Anomaly Detection Benchmark](https://arxiv.org/pdf/2206.09426.pdf)
    1. **None of the benchmarked unsupervised algorithms is statistically better than others**, emphasizing **the importance of algorithm selection** 
    2. With **merely 1% labeled anomalies, most semi-supervised methods can outperform the best unsupervised method**, justifying **the importance of supervision**
    3. In controlled environments, we observe that **best unsupervised methods for specific types of anomalies are even better than semi- and fully-supervised methods**, revealing **the necessity of understanding data characteristics** 
    4. Semi-supervised methods show potential in achieving robustness in noisy and corrupted data possibly due to their efficiency in using labels and feature selection

![](https://i.imgur.com/n5JYEPO.png)

![](https://i.imgur.com/YrrNYYb.png)

:::


### Isolation Forest
[Anomaly Detection with Isolation Forest & Visualization](https://towardsdatascience.com/anomaly-detection-with-isolation-forest-visualization-23cd75c281e2)
### AutoEncoder (Seq-to-Seq)

### PCA

### LSTM



## Interpretation





## Application

### Anomaly detection/ Fault diagnosis



### PHM (prognositic health management)

[PHM step overview](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6024332)



![](https://i.imgur.com/6jazsEh.png)




## Univariate vs. Multivariate Anomaly Detector
- Multivariate Anomaly detectors use time series data with two or more metrics to identify anomalies
    - see how **an outlier detected in one metric relates to other metrics** in the dataset (provide a holistic view of abnormalities from more than one variable)
- Azure Anomaly Detector does not work on images or video frames



- Azure IoT Hub :a communication channel for sending and receiving data between smart devices and cloud
- Azure Blob storage: store massive amounts of unstructured data. It's useful for storing the raw data we'll collect from the Azure IoT Hub