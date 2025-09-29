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


<h2>Video Source Configuration</h2>
<p>
You can specify your video inputs in a <code>.env</code> file. 
These sources can be either <b>local video files</b> or <b>RTSP streams from live cameras</b>.
</p>

<h3>Example (.env)</h3>

<pre>
# Primary video sources (can be RTSP streams or video files)
VIDEO_SOURCE_1=ucf24/videos/Basketball/v_Basketball_g22_c01.mp4

# Example RTSP live camera stream
VIDEO_SOURCE_1=rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1
</pre>

<p>
Once your <code>.env</code> is configured, run YOWOv3 with one of the following modes:
</p>

<h3>🔹 Multistreaming Mode</h3>
<pre>
python main.py -m multistreaming_live -cf config/cf2/ucf_config.yaml
</pre>

<h3>🔹 Single Live Mode (Original YOWOv3)</h3>
<pre>
python main.py -m live -cf config/cf2/ucf_config.yaml
</pre>


<h2>Multistreaming Example</h2>

<table>
  <tr>
    <td align="center">
      <video src="output/v_Basketball_g22_c01_detections_20250929_120428.mp4" width="400" controls></video>
      <p><b>Basketball Stream</b></p>
    </td>
    <td align="center">
      <video src="output/v_Diving_g15_c05_detections_20250929_120428.mp4" width="400" controls></video>
      <p><b>Diving Stream</b></p>
    </td>
  </tr>
</table>
