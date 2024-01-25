
# Video Analytics Using Deep Neural Networks
Design of an efficient deep neural network based on the two stream architecture for the prediction and detection of violent actions in vidoes. Visual surveillance is an essential component for any region where the legally admissible use of surveillance cameras has turned out to be a standard to employ law enforcement and ensure security in a smart city. In real-world surveillance situations, differentiation of violent actions from that of non-violent actions especially aggressive actions that may seem violent but are non violent is still a challenge for computer vision. Several contributions to this domain have been made over the last few years but the problem of effective surveillance still demands researchers attention. The second reason is the lack of an objective and clear definition of violence in a video, as violent events occur infrequently in real world environments and are more diverse when compared to non violent events. To spot these violent activities, an effective technique is required to monitor and evaluate the scenario without human intervention.



## Problem Definition
The goal of this project is to use the two-stream method to develop an accurate and efficient model that can automatically detect hand-to-hand conflict in the live video stream.Hand-to-hand conflicts refer to physical fights that involve direct physical contact between individuals. These conflicts typically occur when two or more individuals engage in combat using their bodies, such as striking, grappling, or wrestling, without the use of weapons.
Real-time live streams refer to the transmission of audio and video captured by cameras and video equipment in real time. These real-time live streams can help us detect violent incidents as they happen, allowing us to intervene in situations rather than post-arrival at the incident scene for analysis and recovery.

## Architecture of the model
The input to the two stream architecture is typically centered around a single frame within the input video, which is directly passed as input into the network’s RBG stream.As input to the other stream, the optical flow of the image is passed .From here, the RGB and Optical streams process the frame and optical flow input using separate convolutional networks with similar structure, the only difference between the respective networks is that the temporal stream is adapted to accept input with a larger number of channels.Once the output of each stream is computed, stream representations are fused together to make a single, spatiotemporal representation that is used for prediction.
![Model Architecture](https://github.com/R3hankhan123/Donnors-Choice/assets/90124241/651e01b5-f766-4473-bee6-78b0b47011c7)

**The parameter of the model is:**
- Loss: Binary Crossentropy
- Optimizer: SGD
- Learning Rate: 0.0000001
- Validation Split: 0.1
- Callbacks: Early Stopping
- monitor: val_loss
- mode: auto
- Batch Size: 32
- Epochs: 200


## Accuracy and Loss
The model was trained in three batches at each batch the accuracy and loss of the model was plotted
**Batch 1**

![](https://github.com/R3hankhan123/Donnors-Choice/assets/90124241/42dfbf64-c91e-4f79-8c9c-e32545ab9a8e)

![](https://github.com/R3hankhan123/Donnors-Choice/assets/90124241/b9b58e43-23ab-4b63-a355-425d45c98a04)

**Batch 2**

![](https://github.com/R3hankhan123/Donnors-Choice/assets/90124241/0b6e739b-fd31-41df-b527-b597ef642fdb)

![](https://github.com/R3hankhan123/Donnors-Choice/assets/90124241/b95745ab-6596-44ad-be49-76ca5b6995ff)

**Batch 3**

![](https://github.com/R3hankhan123/Donnors-Choice/assets/90124241/168dd554-7d7b-44b8-9c3f-00d903dd182a)

![](https://github.com/R3hankhan123/Donnors-Choice/assets/90124241/7ebf514a-8bf5-47a5-b0ad-b52cd39df20d)


## Authors and Contributers

- [Rehan Khan](https://github.com/R3hankhan123)
- [Rahul Kumar](https://github.com/rahulo15)
- [Swaroop S Kammath](https://github.com/cluelescoder)
- [Sahil Tarun Agarwal](https://github.com/Sahil-agarwal-1)
