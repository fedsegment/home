# FedSegment

##  A Federated Learning Framework for Image Segmentation

### Our project is in collaboration with FedML, a research library and benchmark for federated machine learning, developed at USC. Our module will be an open source contribution to FedML. 



![Image](pictures/anim1.gif)


### Overview

Image segmentation is a commonly used technique to partition an image and generate pixel-wise masks for each instance. There are several interesting use cases for segmentation, ranging from medical data analysis to content-based image retrieval. 

However, is it possible to do image segmentation on data we do not have access to?
No, not now! 

Does that limit us to solve problems we care about?
Maybe, yes. 

Let us consider a scenario where top medical institutions have just learned about the novel coronavirus and intend to diagnose the illness using the chest X-ray images of symptomatic patients. Using these images, a segmentation model could be developed in order to facilitate early diagnosis. Unfortunately, this approach is not as straightforward as it sounds. We would inevitably encounter some significant bottlenecks while training the segmentation model. The most noteworthy concern would be data insufficiency due to privacy concerns where hospitals are unwilling to share sensitive patient records, besides lack of adequate computing resources. 

![Image](pictures/pic1.png =200x200)

Thus, the real problem lies in agglomerating all the information in a central location where the model will train. This brings us to the notion of federated learning! With federated learning, we can now take the model to the data instead of bringing the data to the model. 

- 1. A centralized server maintains the global deep neural network model. 
- 2. Each participating institution trains their own copy of the model using their respective datasets. 
- 3. The central server then receives the updated model from each participant and aggregates the contributions.
- 4. The model’s updated parameters are shared with the participants once again, as they continue their local training. 


This agreement of letting developers train the segmentation model on data they cannot have access to is the framework we have built and termed as FedSegment. 


### Approach

We simulate a federated learning architecture on image segmentation by adapting and training DeepLabV3+, a state-of-the-art model for image segmentation, by incorporating the Resnet network as the backbone for extracting feature maps. We develop a data loader for the PASCAL VOC dataset which supports non-I.I.D. distribution to represent real-world, distributed data that naturally tends to be non-I.I.D. Assuming we have virtual clients and a centralized server, the data loader partitions training image dataset in the non-I.I.D format. 

![Image](pictures/anim2.gif)

The server sends initial weights of the model to all the clients in the beginning. The clients start training on their own subset of data and send weights to the server. The server gathers the weights from all the clients and performs aggregation on the weights. The aggregated weights are sent back to the clients and the training continues. This entire loop is called one round. Many such rounds are simulated in order to achieve a decent accuracy. 



