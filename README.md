# Train I3D model on ucf101 or hmdb51 by tensorflow
### This code also for training your own dataset
## How use our code?
### 1.Data_process
>>1>download UCF101 and HMDB51 dataset by yourself<br>
>>2>extract RGB and FLOW frames by [denseFlow_GPU](https://github.com/yangwangx/denseFlow_gpu), such as:<br>
* ~PATH/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/i for all rgb frames<br>
* ~PATH/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/x for all x_flow frames<br>
* ~PATH/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/y for all y_flow frames<br>
>>3>convert images to list for train and test<br>
```linux
cd ./list/ucf_list/
bash ./convert_images_to_list.sh ~path/UCF-101 4
```
* you will get train.list and test.list for your own dataset<br>
* such as: ~PATH/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0<br>
### 2.Train your own dataset(UCF101 as example)
>>1>if you get path errors, please modify by yourself
```linux
cd ./experiments/ucf-101
python train_ucf_rgb.py
python train_ucf_flow.py
```
>>2>argues
* learning_rate: Initial learning rate
* max_steps: Number of steps to run trainer
* batch_size: Batch size
* num_frame_per_clib: Nummber of frames per clib
* crop_size: Crop_size
* classics: The num of class
>>3>models will be stored at ./models, and tensorboard logs will be stored at ./visul_logs

```linux
tensorboard --logdir=~path/I3D/experiments/ucf_101/visual_logs/
```
### 3.Test your own models
>>1>if you get path errors, please modify by yourself
```linux
cd ./experiments/ucf-101
python test_ucf_rgb.py
python test_ucf_flow.py
python test_ucf_rgb+flow.py
```
### 4.Result on my linux
