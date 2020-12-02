<!-- ![Image](pictures/anim1.gif){: id="anim1"} -->
### Overview

Image segmentation is a commonly used technique to partition an image and generate pixel-wise masks for each instance. There are several interesting use cases for segmentation, ranging from medical data analysis to content-based image retrieval. 

However, is it possible to do image segmentation on data we do not have access to?
No, not now! 

Does that limit us to solve problems we care about?
Maybe, yes. 

Let us consider a scenario where top medical institutions have just learned about the novel coronavirus and intend to diagnose the illness using the chest X-ray images of symptomatic patients. Using these images, a segmentation model could be developed in order to facilitate early diagnosis. Unfortunately, this approach is not as straightforward as it sounds. We would inevitably encounter some significant bottlenecks while training the segmentation model. The most noteworthy concern would be data insufficiency due to privacy concerns where hospitals are unwilling to share sensitive patient records, besides lack of adequate computing resources. 

<!-- ![Image](pictures/pic1.png) -->

{% include image.html url="pictures/pic1.png" description="Segmented chest X-ray image" %}{: id="pic1"}

Thus, the real problem lies in agglomerating all the information in a central location where the model will train. This brings us to the notion of federated learning! With federated learning, we can now take the model to the data instead of bringing the data to the model. 

- A centralized server maintains the global deep neural network model. 
- Each participating institution trains their own copy of the model using their respective datasets. 
- The central server then receives the updated model from each participant and aggregates the contributions.
- The model’s updated parameters are shared with the participants once again, as they continue their local training. 


This agreement of letting developers train the segmentation model on data they cannot have access to is the framework we have built and termed as FedSegment. 

<!-- ![Image](pictures/anim2.gif) -->
{% include image.html url="pictures/anim2.gif" description="Animation of FedSegment" %}{: id="anim2"}


### Approach

We simulate a federated learning architecture by adapting and training DeepLabV3+, a state-of-the-art model for image segmentation, and incorporating ResNet as the backbone for extracting feature maps. We develop a data loader for the PASCAL VOC dataset which supports partitioning of the data between our virtual clients in a non-I.I.D manner using Dirichlet, to represent real-world, scattered data. 

The Dirichlet distribution, parameterized by the concentration parameter 𝛼, is a density over a K dimensional vector whose K components are positive and sum up to 1. Dirichlet can support the probabilities of a K-way categorical event. In Federated Learning, we find that the K clients' sample numbers obey the Dirichlet distribution. This Latent Dirichlet Allocation (LDA) method was first proposed by Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification. This can generate non-IIDness with an unbalanced sample number in each label. The figure below illustrates populations drawn for 4 clients from the Dirichlet distribution of the Pascal VOC dataset with 21 classes for different values of 𝛼.

{% include image.html url="pictures/nonIIDGraphComparison.png" description="Non I.I.D comparison with different alpha values (we use alpha=0.5 to achieve good a non-I.I.D distribution)" %}{: id="niid"}

We have 5 worker threads out of which 4 are client-threads managed by their Client Managers and a server thread managed by a Server Manager. Client Managers execute the training process in the order that they receive weights from the Server Manager. In the first round, the Server Manager initializes the model and sends the weights to the Client Manager for each client. The Client Manager then sets up the client worker threads on their respective GPUs for training. Since we are training our architecture on 4 GPUs, each client is assigned one GPU. After completing the specified number of epochs, The Client Managers then send the trained weights to the Server Manager. The Server Manager instantiates the server thread to perform aggregation on model weights received from all the clients by scaling the weights with respect to the size of their local dataset. It then updates the model parameters, performs evaluation on the validation set and sends back the updated model weights to the Client Managers. The next round begins and the process continues for a given number of rounds. We employ the MPI communication protocol for the interactions between the server and its clients. 


### Implementation

<!-- ![Image](pictures/pic2.png) -->
{% include image.html url="pictures/pic2.png" description="Architecture of Deeplabv3+" %}{: id="pic2"}

We perform image segmentation by training the Deeplabv3+ model with ResNet-101 as backbone for feature maps extraction. DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. It uses the Resnet model as a backbone and applies the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network.

We build a segmentation model on top of the open source FedML framework by training the Deeplabv3+ - Resnet 101 model on the PASCAL VOC dataset comprising 10582 augmented training images and 2857 validation images. The centralized training of Deeplabv3+ in the research paper is our reference baseline model which gives a mean Intersection-over-Union (mIoU) of 78.85%. We have achieved an mIoU of 75.57% in the federated setting. Assuming we have 4 virtual clients, we use 4 GPUs where we assign each client one GPU. By conducting several experiments, we have determined the following values for the training parameters to achieve good results: Batch Size: 10, output stride: 16, learning rate: 0.007, optimizer: SGD. We train the model for 60 rounds with 2 epochs per round. We have also enabled a functionality to save checkpoints of the best validation prediction of mIoU by implementing a saver module.    


### Experiments

We adapt the ImageNet-pretrained backbone to the semantic segmentation by applying atrous convolution to extract dense features. For extracting feature maps, we experimented with two backbones; Aligned-Xception and ResNet-101. Motivated by the recent success of depthwise separable convolution, we first explored our experimentation by adapting Xception backbone [2] for extracting feature maps, and reported a mIoU ~10% on validation dataset after 10 rounds. However these numbers surged up when we plugged in the widely used ResNet Backbone. For the same number of rounds, we reported validation mIoU ~60%. Even though Xception backbone is believed to be giving better results with faster computation as per [2], that didn’t turn out to be true during our experimentation. After double checking our implementation, we figured out that the discrepancy was mainly because of the pretrained weights we were using. For the implementation we followed which is a slight modification of Xception backbone called Aligned Xception [5], the pretrained weights we had were not fully trained. And we could not find fully pretrained weights for the exact implementation. Hence we decided to stick with ResNet Backbone for which we had fully pretrained weights already available.

In addition to experimenting with two different backbones, we also experimented with different output strides. The output stride is defined as the ratio of input image spatial resolution to final output resolution. For the task of semantic segmentation, one can adopt output stride = 16 (or 8) for denser feature extraction. Theoretically, lowering the output stride improves the performance marginally, however it adds a lot of extra computational complexity and hence using output stride = 16 strikes the best trade-off between speed and accuracy. With the limited resources we had, we could not only train our models upto round 15 for output stride = 8 as reported in the table below. 

| Backbone | Batch Size | Epochs | Rounds | Learning Rate | Testing Accuracy | Testing Loss | mIoU | fWIoU |
|---|---|---|---|---|---|---|---|---|
| ResNet-101 | 6 | 2 | 15 | 0.007 | 0.899 | 0.0083 | 0.655 | 0.828 | 
| ResNet-101 | 10 | 2 | 60 | 0.007 | 0.924 | 0.0024 | 0.755 | 0.863 | 
| Xception | 6 | 2 | 60 | 0.007 | 0.6445 | 0.0195 | 0.048 | 0.419 | 


### Results

As shown in the table above the model architecture which gave us the best results was with output_stride = 16, batch_size = 10 with a learning rate of 0.007 while fine-tuning the backbone and 0.07 for all the remaining layers. We trained the model only until 60 rounds as the model reaches saturation stage and the validation accuracy does not improve post that. At each round, the model was trained for 2 epochs locally by all four clients. The results for the best performing federated architecture for our model are as shown in the graph below.

<!-- ![Image](pictures/finl_graphs.png) -->
{% include image.html url="pictures/finl_graphs.png" description="Final Graphs" %}{: id="final-graphs"}

{% include image.html url="pictures/final_inference_visualization.png" description="Final Visualization" %}{: id="final-visualization"}

### Challenges

We identify three major challenges for using the above mentioned segmentation method in a real world setting:
    
1. Computational Complexity:
Since image segmentation is a computationally expensive task, hence edge devices with resource constraint will always face challenges for our method. There are few ways to reduce the computational complexity for e.g reducing the complexity of training by freezing the backbone or keeping a higher output_stride but all of that will come at the cost of losing some accuracy. So, there is again a tradeoff to make between accuracy and computational complexity.
2. Statistical Heterogeneity:
Since devices frequently generate and collect data in a non-identically distributed manner across the network, hence the number of data points across edge devices may vary significantly and this data generation paradigm violates frequently-used independent and identically distributed (I.I.D) assumptions in distributed optimization. To mitigate this problem, the recommended approach is to simulate the non-IID setting as closely as possible while training the model so the model learns to generalize and adapts to this nature of data generation.
3. Privacy Concerns:
Even though the entire concept of federated learning is to protect the privacy of data, privacy is still a major concern in federated learning applications including image segmentation. Even though federated learning makes a step towards protecting data generated on each device by sharing model updates, e.g, gradient information, instead of raw data. However, communicating model updates throughout the training process can still reveal sensitive information either to a third-party or to the central server. From the shift in gradient at each state, there are some statistical ways to infer a good amount of information about the data points that are supposed to be private. While recent methods aim to enhance the privacy of federated learning using tools such as multipart computation or differential privacy, these approaches often provide privacy at the cost of reduced model performance or system efficiency. Hence there is a need to understand and balance these tradeoffs while using a federated segmentation approach.


### Future Work

There are two things that we intend to work on:
1. Improving accuracy of the model - 
In addition to hyper parameter tuning, we plan to work on improving the model accuracy by experimenting with lesser output strides. As mentioned above, the reduction of output strides comes at a huge cost of computational complexity. To keep the computational complexity in check, we can freeze the backbone and try experimenting with output stride of 8 or 4. Also, plugging a better loss function (Dice + Focal Loss) instead of Cross-Entropy Loss may be helpful. Lastly, pre-training the model on Coco Dataset and fine-tuning it on Pascal can give better results because Pascal has a small dataset which further gets splitted into n number of clients and so there is a good chance for model to overfit when data is splitted amongst more number of clients.
2. Integrating other popular segmentation models - 
We also intend to explore and incorporate alternate backbones to train the DeepLabV3+ model - such as Xception, MobileNet, etc., which have pretrained models resulting in a state of the art accuracy. We already tried employing the Xception Backbone but the results weren’t impressive because of the reasons mentioned earlier. However, with a fully trained Xception, we expect to get similar or better results than Resnet Backbone that we currently have. Moreover, our approach is limited to using the ResNet backbone. We also plan on extending our current work and encompass additional segmentation models such as EfficientFCN or BlendMask under the FedSegment umbrella.

### Resources

1. https://fedml.ai/
2. https://arxiv.org/abs/2007.13518
3. https://arxiv.org/pdf/1802.02611.pdf
4. https://wandb.ai/elliebababa/fedml/runs/5ykbr3ul
5. https://github.com/jfzhang95/pytorch-deeplab-xception

