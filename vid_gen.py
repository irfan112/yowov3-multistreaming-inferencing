'''
ucf24 dataset download 
https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view?pli=1
generate single video for each class ----> Testing
'''


import os
import subprocess

root_dir = "ucf24/rgb-images"
output_dir = "ucf24/videos"

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    print(f"folder: {class_name}")
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    print("seq_name: ", os.listdir(class_dir)[0])
    seq_dir = os.path.join(class_dir, os.listdir(class_dir)[0])
    if not os.path.isdir(seq_dir):
        continue

    out_video = os.path.join(class_output_dir, f"{os.listdir(class_dir)[0]}.mp4")
    print(f"Generating video: {out_video}")

    # Run ffmpeg
    cmd = [
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(seq_dir, "%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_video
    ]
    subprocess.run(cmd, check=True)
