import numpy as np
import gradio as gr
from predict import Predictor
import torch 
from PIL import Image


def predict_pose(input_img):
    # input_img = Image.fromarray(input_img)
    # input_img.save("upload.jpg")

    with torch.no_grad():
        result_img = obj.predict_pose_image(input_img, num_poses=1, sample_algorithm = 'ddim',  nsteps = 50)
    torch.cuda.empty_cache()  # 释放显存
    return result_img

def predict_appearance(input_img):
    # input_img = Image.fromarray(input_img)
    # input_img.save("upload.jpg")

    with torch.no_grad():
        ref_img = "data/deepfashion_256x256/target_edits/reference_img_3.png"
        ref_mask = "data/deepfashion_256x256/target_mask/upper/reference_mask_3.png"
        ref_pose = "data/deepfashion_256x256/target_pose/reference_pose_3.npy"
        result_img = obj.predict_appearance_image(image = input_img, ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)
    torch.cuda.empty_cache()  # 释放显存
    return result_img

def PIDM(input_img,mode):
    if mode == None:
        raise gr.Error("Please select a mode!")
    if mode == "Pose Prediction":
        result_img = predict_pose(input_img)
        return result_img
    elif mode == "Appearance Prediction":
        result_img = predict_appearance(input_img)
        return result_img
    else:
        raise gr.Error("Error!")
        
    # sepia_filter = np.array([
    #     [0.393, 0.769, 0.189], 
    #     [0.349, 0.686, 0.168], 
    #     [0.272, 0.534, 0.131]
    # ])
    # sepia_img = input_img.dot(sepia_filter.T)
    # sepia_img /= sepia_img.max()
    
obj = Predictor()
demo = gr.Interface(PIDM, 
        inputs=[gr.Image(),gr.Radio(["Pose Prediction", "Appearance Prediction"])],  
        outputs=[gr.Image()],
        title="Person Image Synthesis via Denoising Diffusion Model",
    )
demo.launch(share=True,auth=("username", "password"),server_name="10.40.18.41")   
