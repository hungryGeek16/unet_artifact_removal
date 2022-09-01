import cv2
import argparse
import os
from PIL import Image 
import numpy as np
from model import *

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="", help="path to ckpt")
args = parser.parse_args()

# Set env varibales, _-outputs is necessary for infer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
checkpoint_dir = "./__outputs/checkpoints/"
summary_dir = "./__outputs/summary/"


# Predict image function for patchwise inference.
def predict(image, model):
  image = np.array(image)
  ts = np.empty([image.shape[0], image.shape[1], 3], dtype=np.uint8)
  image = normalize(image)
  for x in range(0,image.shape[0],48):
    for y in range(0,image.shape[1],48):
      im = image[x:x+48,y:y+48]
      im = np.expand_dims(im.astype("float32"), axis=0)
      outs = model.generator(im)
      output = denormalize(outs[-1])
      ex = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
      ts[x:x+48, y:y+48, :] = output
  return ts

# Initialize model
model = Model_Train(summary_dir, checkpoint_dir, 0.001, 0.00001)
model.ckpt.restore(args.ckpt_path)
for i in os.listdir("dataset/validation/down_grade/"):
  image = Image.open("dataset/validation/down_grade/"+i).convert('RGB')
  image_clean = Image.open("dataset/validation/clean/"+i).convert('RGB')
  image_clean = np.array(image_clean)
  if image_clean.shape[0] % 48 != 0 or image_clean.shape[1] % 48 != 0:
    continue
  output = predict(image, model)
  im = Image.fromarray(output)
  print("\n PSNR for original/reconstructed:"+i,tf.image.psnr(image_clean, output, 255))
  print("\n SSIM for original/reconstructed:"+i,tf.image.ssim(tf.convert_to_tensor(image_clean),tf.convert_to_tensor(output),255))
  im.save("preds_"+i)
  image.save("dg_"+i)