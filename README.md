# TartanVO Challenge
My solution for the [TartanVO VSLAM challenge Mono Track](https://www.aicrowd.com/challenges/tartanair-visual-slam-mono-track). Incorporates an end-to-end RCNN solution for position and pose estimation using a streaming-training paradigm in order to cope with the 3TB training set.

The work here is inspired by [this work](https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf).

## Network Architecture
![tartan vo network iteration one](https://github.com/bakshienator77/tartanVO_challenge_solution/blob/master/images/Tartan_block_diagram%20copy?raw=true)

- The input to the model is a pair of images at half resolution than what is available
- The model predicts the change in position and orientation (expressed as a unit quaternion) in NED frame
- The high number of nodes in the input layer to the LSTM is alarming, yes. This is a first draft of the model, the architecture needs improvement.
- LeakyRelU prevents dead nodes and overfitting
- BN should make training faster

## Streaming-Training
Since this dataset is massive I found it quite prohibitive to use as:
- I can't download it to my local machine
- It costs a lot to store so much data on an SSD in the cloud
- I didn't want to wait for it all to be downloaded before I could start training
- Cloud GPUs are costly enough without me having to worry about leaving my data on the cloud overnight too

Therefore the simple solution implemented here aims for memory efficiency with minimal downloading of data. *This does not guarantee the most efficient training regime*.

### Training in trajectory lengths
The dataloader is implemented in the `deepvo.py` file and is called [VisualOdometryDataLoader](https://github.com/bakshienator77/tartanVO_challenge_solution/blob/master/deepvo.py#L569). The Dataloader is passed a list of lists, where:
- each entry in the upper list corresponds to one trajectory from the training data (defaults to left image since we are training a monocular model)
- each sublist is a list of paths (str) to the images in that trajectory

In order to make training memory efficient and because we a using a sequence model in our architecture (cannot backpropagate infitely back in time) we set a hyperparameter known as `trajectory_length` which defines how far back we back propagate.

At any given time we keep ```batch_size * (trajectory_length + 1)``` images in memory. We download images sequentially such that, barring the first step of training, only `batch_size` number of images are downloaded in each subsequent step and each downloaded image is *used* approximately `trajectory_length` number of times.

Concretely, if we have a `batch_size` of 8, each trajectory in the batch is 100 frames long and our `trajectory_length` is set to 10. This batch will further be divided into ```#frames - trajectory_length = 90``` mini-batches, in each mini-batch the the frames are fed in pair-wise sequentially starting from the 0th and upto the 10th frame followed by an optimizer step.

### The wonder that is Azure
This solution is extremely sensitive to network failures and, as I soon learned, Azure's *feature* of infinite timeout. Essentially what happens is that when a request to download a blob is sent from azure storage, we can specifiy a timeout parameter, however if, for any reason, this request doesn't reach azure it also doesn't recieve the timeout value and we are just stuck in limbo forever. The surprising part is that this timeout interrupt is not implemented on the sender-side and is instead on the receiver-side. Since I implemented a checkpoint only at the end of each batch, that means the training progress was lost. It was also a real hassle as it meant I had to constantly monitor it as this happened several times a day.

Enter `timeout-decorator`. This little package completed the functionality I was looking for as I could not set the timeout to something aggressive like 1 sec and at the first sight of trouble, retry any request. Training was smooth after this.

## Improvements possible

- My logging is quite lousy
- Architecture definitely needs sprucing up
- Parallel gpus or tpus
- Training longer
- Improved training regime that chooses random starting frames trading away some memory efficiency for better training
- Training for much longer than I'd care to pay for out of pocket
