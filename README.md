# unet_artifact_removal
A CNN-based model that could remove jpeg artifacts from a given jpeg image. 

# Training and Inference:

* Tensorflow version 2.1 is required to execute training and inference of the model.

* Installation of necessay libraries can be done using the command given below:

```bash
pip install -r requirements.txt
```

* Download the dataset for training and testing by executing the command given below:

```bash
gdown 1Bppd8WW2N7Ji2BVizDINBkbuaZg9lkUY
unzip jpeg_dataset.zip
rm jpeg_dataset.zip
```
* Please make sure that the dataset is present in the same directory as the repository.

* For training, execute command given below:

```bash
python train.py --steps 20000
```

* This will train the model for 20,000 steps and will output a folder named "UNET_month_day_hour_minute_second".
* Folder named outputs would be created while training that has model's output results at every step.
* Another folder named __outputs will be given out that has checkpoint and summary details, please do not delete this folder.

* To check model's performance on some image from the dataset, please execute the command given below:

```bash
python infer.py --ckpt_path path/to/UNET_month_day_hour_minute_second/ckpt-20000
```
* This will output two image files, one would be the downgraded image and another would be the artifact removed image.

