import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob
import threading
import queue
from collections import deque
from math import sqrt
from cus_datasets.build_dataset import build_dataset
# from utils.bbox_yowo import draw_bounding_box
from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from PIL import Image
from utils.flops import get_info
from RTSP.live_utils import live_transform
import logging

from dotenv import load_dotenv
load_dotenv(override=True)


class VideoStreamThread(threading.Thread):
    def __init__(self, stream_id, video_source, config, yowo_model, mapping, \
                 max_fps=20, viz_width=720, viz_height=448, save_video=False):

        threading.Thread.__init__(self)
        self.stream_id = stream_id
        self.video_source = video_source
        self.config = config
        self.yowo_model = yowo_model
        self.mapping = mapping
        self.max_fps = max_fps
        self.frame_interval = 1.0 / max_fps 
        if 'rtsp://' in video_source:
            self.camera_name = f"Camera_{stream_id}"
        else:
            self.camera_name = f"Stream_{stream_id}_{os.path.basename(video_source).split('.')[0]}"
        
        self.viz_width = viz_width
        self.viz_height = viz_height
        
        self.cap = cv2.VideoCapture(video_source)
        
        if video_source.startswith('rtsp://'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
            self.cap.set(cv2.CAP_PROP_FPS, max_fps)   
        
        self.transform = live_transform(config['img_size'])
        self.frame_buffer = deque(maxlen=16)  
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)  
        
        # RTSP-specific parameters
        self.is_rtsp = video_source.startswith('rtsp://')
        self.last_successful_frame = None
        self.connection_retry_count = 0
        self.max_retry_attempts = 5
        self.frame_timeout = 2.0  
        
        self.scale_x = self.viz_width / config['img_size']
        self.scale_y = self.viz_height / config['img_size']

        self.viz_width = viz_width
        self.viz_height = viz_height

        self.save_video = save_video
        self.video_writer = None
        if self.save_video:
            output_dir = "output"  # Change this to your preferred directory
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if 'rtsp://' in video_source:
                output_filename = f"{output_dir}/camera_{stream_id}_detections_{timestamp}.mp4"
            else:
                base_name = os.path.basename(video_source).split('.')[0]
                output_filename = f"{output_dir}/{base_name}_detections_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_filename, fourcc, max_fps, (viz_width, viz_height))
            print(f"Saving processed video for stream {stream_id} to: {output_filename}")

        
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video source {video_source}")
    
    def reconnect_rtsp(self):
        """Reconnect to RTSP stream"""
        print(f"Attempting to reconnect to stream {self.stream_id}...")
        
        if self.cap.isOpened():
            self.cap.release()
        
        time.sleep(1.0)
        
        self.cap = cv2.VideoCapture(self.video_source)
        
        if self.video_source.startswith('rtsp://'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        
        self.connection_retry_count += 1
        
        if self.cap.isOpened():
            print(f"Successfully reconnected to stream {self.stream_id}")
            self.connection_retry_count = 0  
            return True
        else:
            print(f"Failed to reconnect to stream {self.stream_id} (attempt {self.connection_retry_count})")
            return False
    
    def read_frame_with_timeout(self):
        """Read frame with timeout handling for RTSP"""
        if not self.is_rtsp:
            return self.cap.read()
        
        start_time = time.time()
        ret = False
        frame = None
        
        try:
            ret, frame = self.cap.read()
            
            if time.time() - start_time > self.frame_timeout:
                print(f"Frame reading timeout for stream {self.stream_id}")
                ret = False
                frame = None
                
        except Exception as e:
            print(f"Error reading frame from stream {self.stream_id}: {e}")
            ret = False
            frame = None
        
        return ret, frame
    
    def scale_bounding_boxes(self, outputs):
        """Scale bounding boxes from model size to visualization size"""
        if outputs is None:
            return None
        
        scaled_outputs = outputs.clone()
        scaled_outputs[:, [0, 2]] *= self.scale_x
        scaled_outputs[:, [1, 3]] *= self.scale_y
        
        return scaled_outputs
    
    def draw_yowo_bounding_box(self, image, boxes, class_ids, scores, mapping):
        """Draw YOWO bounding boxes"""
        if boxes is None or len(boxes) == 0:
            return image
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            class_id = int(class_ids[i])
            score = float(scores[i])
            
            x1 = max(0, min(x1, self.viz_width - 1))
            y1 = max(0, min(y1, self.viz_height - 1))
            x2 = max(0, min(x2, self.viz_width - 1))
            y2 = max(0, min(y2, self.viz_height - 1))
            
            class_name = mapping.get(class_id, f"Action_{class_id}")
            
            yowo_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                          (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
            color = yowo_colors[class_id % len(yowo_colors)]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Create label
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image

    def run(self):
        frame_count = 0
        last_frame_time = time.time()
        consecutive_failures = 0
        last_frame_timestamp = time.time()
        
        while self.running:
            try:
                # Read frame with timeout handling
                ret, frame = self.read_frame_with_timeout()
                
                # Handle RTSP connection issues
                if not ret or frame is None:
                    consecutive_failures += 1
                    
                    if self.is_rtsp and consecutive_failures > 10:
                        if self.connection_retry_count < self.max_retry_attempts:
                            if self.reconnect_rtsp():
                                consecutive_failures = 0
                                continue
                        else:
                            print(f"Max retry attempts reached for stream {self.stream_id}")
                            time.sleep(5.0)  # Wait before continuing
                            self.connection_retry_count = 0
                            continue
                    else:
                        # Use last successful frame if available
                        if self.last_successful_frame is not None:
                            frame = self.last_successful_frame.copy()
                            ret = True
                        else:
                            time.sleep(0.1)
                            continue
                else:
                    # Successful frame read
                    consecutive_failures = 0
                    self.last_successful_frame = frame.copy()
                    last_frame_timestamp = time.time()
                
                if self.is_rtsp:
                    # For RTSP, skip more aggressively to reduce latency
                    frame_skip = max(1, int(30 / self.max_fps))  # Assume 30fps source
                    if frame_count % frame_skip != 0:
                        frame_count += 1
                        continue

                # FPS control - more relaxed for RTSP
                current_time = time.time()
                if self.is_rtsp:
                    # For RTSP, prioritize smooth playback over exact FPS
                    min_interval = self.frame_interval * 0.8  # Allow 20% faster
                    elapsed_time = current_time - last_frame_time
                    if elapsed_time < min_interval:
                        time.sleep(min_interval - elapsed_time)
                else:
                    # Original FPS control for video files
                    elapsed_time = current_time - last_frame_time
                    if elapsed_time < self.frame_interval:
                        time.sleep(self.frame_interval - elapsed_time)
                
                last_frame_time = time.time()
                
                # Process frame for YOWO model input
                origin_image = Image.fromarray(frame)
                transformed_frame = self.transform(origin_image)
                
                # Add to buffer
                self.frame_buffer.append(transformed_frame)
                
                # Prepare visualization frame with custom dimensions
                viz_frame = cv2.resize(frame, (self.viz_width, self.viz_height))
                
                yowo_detections_count = 0
                
                if len(self.frame_buffer) == 16:
                    clip = torch.stack(list(self.frame_buffer), 0).permute(1, 0, 2, 3).contiguous()
                    clip = clip.unsqueeze(0).to("cuda")
                    
                    with torch.no_grad():
                        yowo_outputs = self.yowo_model(clip)
                        yowo_outputs = non_max_suppression(yowo_outputs, conf_threshold=0.5, iou_threshold=0.5)[0]
                    
                    if yowo_outputs is not None:
                        yowo_detections_count = len(yowo_outputs)
                        scaled_outputs = self.scale_bounding_boxes(yowo_outputs)

                        # Draw YOWO bounding boxes
                        viz_frame = self.draw_yowo_bounding_box(
                            viz_frame, 
                            scaled_outputs[:, :4].cpu().numpy(), 
                            scaled_outputs[:, 5].cpu().numpy(), 
                            scaled_outputs[:, 4].cpu().numpy(), 
                            self.mapping
                        )
                
                # Add stream status info
                if self.is_rtsp:
                    latency = time.time() - last_frame_timestamp
                    status = "LIVE" if consecutive_failures == 0 else f"RETRY({consecutive_failures})"
                    info_text = f"Stream {self.stream_id} [{status}] Latency: {latency:.1f}s | Detections: {yowo_detections_count}"
                else:
                    info_text = f"Stream {self.stream_id} | Detections: {yowo_detections_count}"
                
                cv2.putText(viz_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(viz_frame, (5, 5), (text_size[0] + 15, 40), (0, 0, 0), -1)
                cv2.putText(viz_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                try:
                    if self.is_rtsp:
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                break
                    
                    if self.save_video and self.video_writer is not None:
                        self.video_writer.write(viz_frame)
                    
                    self.frame_queue.put((viz_frame, time.time()), block=False)
                except queue.Full:
                    pass
                
                frame_count += 1
                
                # Memory cleanup
                del frame
                if 'clip' in locals():
                    del clip
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing frame in stream {self.stream_id}: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
                continue
    
    def get_latest_frame(self):
        """Get the latest processed frame from the queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None, None
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Video saved for stream {self.stream_id}")
            
        if self.cap.isOpened():
            self.cap.release()


class DisplayManager:
    def __init__(self, stream_threads, max_display_fps=15, viz_width=720, viz_height=448):
        self.stream_threads = stream_threads
        self.display_interval = 1.0 / max_display_fps
        self.last_display_time = time.time()
        self.viz_width = viz_width
        self.viz_height = viz_height
        
        for i in range(len(stream_threads)):
            window_name = f'Stream {i} - YOWO Detection ({viz_width}x{viz_height})'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, viz_width, viz_height)
    
    def update_display(self):
        """Update display for all streams with FPS control"""
        current_time = time.time()
        
        # Control display FPS
        if current_time - self.last_display_time < self.display_interval:
            return False
        
        self.last_display_time = current_time
        
        for i, thread in enumerate(self.stream_threads):
            if thread.is_alive():
                frame, frame_time = thread.get_latest_frame()
                if frame is not None:
                    window_name = f'Stream {i} - YOWO Detection ({self.viz_width}x{self.viz_height})'
                    cv2.imshow(window_name, frame)
        
        return True
    
    def cleanup(self):
        """Clean up display windows"""
        cv2.destroyAllWindows()


def detect(config, viz_width=720, viz_height=448):
    yowo_model = build_yowov3(config)  
    get_info(config, yowo_model)
    yowo_model.to("cuda")
    yowo_model.eval()
    mapping = config['idx2name']

    # Get video sources from environment or use defaults

    video_sources = []
    i = 1
    while True:
        source = os.getenv(f"VIDEO_SOURCE_{i}")
        if source is None:
            break
        video_sources.append(source)
        i += 1

    # if not video_sources:
    #     # Default video sources if none specified - can be RTSP or video files
    #     video_sources = [
    #         "ucf24/videos/Basketball/v_Basketball_g22_c01.mp4",
    #         "ucf24/videos/Diving/v_Diving_g15_c05.mp4",
    #         # Can also be RTSP streams like:
    #         # "rtsp://admin:Test@123%24@192.168.1.162:554/cam/realmonitor?channel=1&subtype=1",
    #     ]
    
    max_fps = int(os.getenv("MAX_FPS", "15"))
    
    print(f"YOWO Multi-Stream Detection System")
    print(f"Visualization window size: {viz_width}x{viz_height}")
    print(f"Model input size: {config['img_size']}x{config['img_size']}")
    print("Supports both RTSP streams and video files (.mp4)")
    
    stream_threads = []
    
    try:
        for i, source in enumerate(video_sources):
            
            save_video = os.getenv("SAVE_OUTPUT_VIDEO", "false").lower() == "true"
            thread = VideoStreamThread(i, source, config, yowo_model, mapping,
                         max_fps, viz_width, viz_height, save_video)
            

            stream_threads.append(thread)
            thread.start()
            print(f"Started thread for stream {i}: {source}")
        
        display_manager = DisplayManager(stream_threads, max_display_fps=12,
                                       viz_width=viz_width, viz_height=viz_height)
        
        print(f"Processing {len(stream_threads)} streams...")
        print("Press 'q' to quit")
        
        # Main display loop
        while True:
            # Update display
            display_manager.update_display()
            
            # Check for quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            
            # Check if all threads are still alive
            active_threads = [t for t in stream_threads if t.is_alive()]
            if not active_threads:
                print("All video streams ended")
                break
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Stop all threads
        print("Stopping all threads...")
        for thread in stream_threads:
            thread.stop()
        
        # Wait for all threads to finish
        for thread in stream_threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                print(f"Warning: Thread {thread.stream_id} did not stop gracefully")
        
        # Clean up display
        display_manager.cleanup()
        print("Cleanup completed")


if __name__ == "__main__":
    config = build_config()
    
    # You can customize the visualization window size here
    # Example: detect(config, viz_width=1280, viz_height=720)  # For 16:9 aspect ratio
    # Example: detect(config, viz_width=960, viz_height=540)   # For smaller 16:9
    # Default: detect(config, viz_width=720, viz_height=448)   # For 16:10 aspect ratio
    
    detect(config, viz_width=720, viz_height=448)