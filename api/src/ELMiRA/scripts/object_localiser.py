#!/usr/bin/env python3

# ADDED: extra imports for timing, saving, and logging JSON
import os           # ADDED
import time         # ADDED
import json         # ADDED

from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from transformers.image_utils import ImageFeatureExtractionMixin

import cv_bridge
import rospy
import sensor_msgs.msg

from elmira.msg import DetectedObject
from elmira.srv import DetectObjects, DetectObjectsResponse


class OWLv2(nn.Module):
    def __init__(self, owl_version="owlv2", score_threshold=0.05):
        super(OWLv2, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if owl_version == "owl-vit":
            self.model = OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-base-patch32"
            )
            self.processor = OwlViTProcessor.from_pretrained(
                "google/owlvit-base-patch32"
            )
        else:
            self.processor = Owlv2Processor.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
            self.model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
        self.mixin = ImageFeatureExtractionMixin()
        self.score_threshold = score_threshold

    def forward(self, image, text_query):
        # set the model in evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # open and prepare the image
        if image.height != image.width:
            image = self.mixin.resize(image, min(image.height, image.width))
        inputs = self.processor(text=text_query, images=image, return_tensors="pt").to(
            self.device
        )

        # get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values)
        valid_indices = torch.where(scores >= self.score_threshold)

        # get prediction labels and boundary boxes
        scores = scores[valid_indices].cpu().detach().numpy()
        labels = logits.indices[valid_indices].cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0][valid_indices].cpu().detach().numpy()
        return scores, boxes, labels

    def plot_predictions(self, image, text_queries, scores, boxes, labels):
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", 24)
        label_offset = 5
        label_padding = 5

        for score, box, label in zip(scores, boxes, labels):
            cx, cy, w, h = box
            cx *= image.width
            cy *= image.height
            w *= image.width
            h *= image.height
            # draw object bounding box
            draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], width=3)
            # create text with score and label
            label_text = f"{text_queries[label]}: {score:1.2f}"
            # determine bbox of label text
            textbbox = draw.textbbox(
                [cx - w / 2 + label_padding, cy + h / 2 + label_offset + label_padding],
                label_text,
                font=font,
                anchor="lt",
            )
            # draw textbbox
            draw.rectangle(
                [
                    textbbox[0] - label_padding,
                    textbbox[1] - label_padding,
                    textbbox[2] + label_padding,
                    textbbox[3] + label_padding,
                ],
                fill="white",
                outline="red",
                width=3,
            )
            # add label text
            draw.text(
                [cx - w / 2 + label_padding, cy + h / 2 + label_offset + label_padding],
                label_text,
                fill="red",
                font=font,
                anchor="lt",
            )
        return image


class OWLv2Server:
    def __init__(self):
        rospy.init_node("owlv2_server")

        # ADDED: measure and log model load time (without touching detection logic)
        t0_load = time.perf_counter()  # ADDED
        self.owlv2 = OWLv2("owlv2")
        self.model_load_ms = (time.perf_counter() - t0_load) * 1000.0  # ADDED
        rospy.loginfo("OWLv2 loaded in %.1f ms", self.model_load_ms)  # ADDED

        self.bridge = cv_bridge.CvBridge()
        rospy.Service("object_detector", DetectObjects, self.detection_request_handler)

        # ADDED: image saving controls (like F-VLM)
        self.save_debug_images   = bool(rospy.get_param("~save_debug_images", True))        # ADDED
        self.save_only_on_det    = bool(rospy.get_param("~save_only_on_detection", False))  # ADDED
        self.save_dir            = rospy.get_param("~save_dir", "/home/hb/elmira_captures") # ADDED
        if self.save_debug_images:                                                          # ADDED
            try: os.makedirs(self.save_dir, exist_ok=True)                                  # ADDED
            except Exception as e:                                                          # ADDED
                rospy.logwarn("Could not create %s: %s", self.save_dir, e)                  # ADDED
                self.save_debug_images = False                                              # ADDED

        # kept as in your original
        self.debug_pub = rospy.Publisher(
            "owlv2_server/result_image",
            sensor_msgs.msg.Image,
            latch=True,
            queue_size=1,
        )
        rospy.loginfo("OWLv2 started successfully")
        rospy.spin()

    def detection_request_handler(self, request):
        req_t0 = time.perf_counter()  # ADDED: total request timer

        # get latest camera image
        t0 = time.perf_counter()  # ADDED
        img_msg = rospy.wait_for_message(
            "/nico/vision/right",
            sensor_msgs.msg.Image,
        )
        wait_for_image_ms = (time.perf_counter() - t0) * 1000.0  # ADDED

        # convert message to PIL image
        t0 = time.perf_counter()  # ADDED
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
        image = Image.fromarray(cv_image)
        decode_ms = (time.perf_counter() - t0) * 1000.0  # ADDED

        # detect objects (UNCHANGED detection path)
        t0 = time.perf_counter()  # ADDED
        scores, boxes, labels = self.owlv2(image, request.texts)
        detect_ms = (time.perf_counter() - t0) * 1000.0  # ADDED

        # visualize detection (UNCHANGED)
        t0 = time.perf_counter()  # ADDED
        vis_image = self.owlv2.plot_predictions(image.copy(), request.texts, scores, boxes, labels)
        draw_ms = (time.perf_counter() - t0) * 1000.0  # ADDED

        # publish debug image (UNCHANGED)
        t0 = time.perf_counter()  # ADDED
        debug_img_msg = self.bridge.cv2_to_imgmsg(np.array(vis_image), "rgb8")
        debug_img_msg.header.stamp = rospy.Time.now()
        self.debug_pub.publish(debug_img_msg)
        publish_ms = (time.perf_counter() - t0) * 1000.0  # ADDED

        # ADDED: save RAW + OVERLAY images (like F-VLM)
        if self.save_debug_images and (len(labels) > 0 or not self.save_only_on_det):  # ADDED
            try:                                                                         # ADDED
                ts = time.strftime("%Y%m%d_%H%M%S")                                      # ADDED
                raw_path = os.path.join(self.save_dir, f"raw_{ts}.jpg")                  # ADDED
                ovl_path = os.path.join(self.save_dir, f"detect_{ts}.jpg")               # ADDED
                Image.fromarray(cv_image).save(raw_path, format="JPEG")                  # ADDED
                vis_image.save(ovl_path, format="JPEG")                                  # ADDED
                rospy.loginfo("Saved images:\n  %s\n  %s", raw_path, ovl_path)           # ADDED
            except Exception as e:                                                       # ADDED
                rospy.logwarn("Failed to save debug images: %s", e)                      # ADDED

        # ADDED: print timing summary to terminal (JSON), no change to service response
        total_ms = (time.perf_counter() - req_t0) * 1000.0  # ADDED
        timings = {                                         # ADDED
            "model_load_ms": round(self.model_load_ms, 1),
            "wait_for_image_ms": round(wait_for_image_ms, 1),
            "decode_ms": round(decode_ms, 1),
            "detect_ms": round(detect_ms, 1),
            "draw_ms": round(draw_ms, 1),
            "publish_ms": round(publish_ms, 1),
            "request_total_ms": round(total_ms, 1),
            "returned_objects": int(len(labels)),
        }
        rospy.loginfo("TIMINGS %s", json.dumps(timings, sort_keys=True))  # ADDED

        # return response with detected objects (UNCHANGED)
        return DetectObjectsResponse(
            [
                DetectedObject(request.texts[labels[i]], scores[i], *boxes[i])
                for i in range(len(labels))
            ]
        )


if __name__ == "__main__":
    OWLv2Server()

