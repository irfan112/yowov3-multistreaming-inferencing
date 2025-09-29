# yowov3-multistreaming 
YOWOv3(Spatio Temporal Action Detection task) using (UCF101-24) dataset. The repo is extension of https://github.com/Hope1337/YOWOv3, https://arxiv.org/pdf/2408.02623


<h2>Environment Setup</h2>

<p><b>Clone this repository:</b></p>
<pre>
git clone https://github.com/irfan112/yowov3-multistreaming-inferencing.git
</pre>

<p>Use <code>Python 3.8</code> or <code>Python 3.9</code>, and then install the dependencies:</p>
<pre>
pip install -r requirements.txt
</pre>

<pre>
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
</pre>


<h2>Datasets</h2>

<h3>UCF101-24</h3>
<p>
Download from: 
<a href="https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view" target="_blank">
Google Drive Link
</a>
</p>


<h2>Pretrained Weights & Checkpoints</h2>
<p>
To train or evaluate <b>YOWO (I3D / ResNet)</b>, you need to download the pretrained weights 
and checkpoints provided here:
</p>
<p>
<a href="https://drive.google.com/drive/folders/1TYrbwfOy9eRQhNQhOk4JJnd4N-rcKReV?usp=sharing" target="_blank">
Google Drive - YOWO (I3D / ResNet) Checkpoints
</a>
</p>

<p>
After downloading, place the files into the corresponding <code>weights/</code> or <code>checkpoints/</code> 
folder in this repository (create them if they don’t exist).
</p>

<pre>
yowov3-multistreaming-inferencing/
│── weights/
│   ├── yowo_i3d.pth
│   ├── yowo_resnet.pth
│── checkpoints/
│   ├── checkpoint_epoch_XX.pth
</pre>

