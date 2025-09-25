#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json
import numpy as np
from PIL import Image
import cv2
import rospy
import cv_bridge
import sensor_msgs.msg
from std_msgs.msg import String

from elmira.msg import DetectedObject
from elmira.srv import DetectObjects, DetectObjectsResponse

# ---------------- utilities ----------------
def _flatten_dict(tree, parent_key=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            nk = f"{parent_key}/{k}" if parent_key else str(k)
            out.update(_flatten_dict(v, nk))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            nk = f"{parent_key}[{i}]"
            out.update(_flatten_dict(v, nk))
    else:
        out[parent_key] = tree
    return out

def _to_numpy(x):
    try:
        import tensorflow as tf
    except Exception:
        tf = None
    if isinstance(x, np.ndarray):
        return x
    if tf is not None and hasattr(tf, "Tensor") and isinstance(x, tf.Tensor):
        try:
            return x.numpy()
        except Exception:
            pass
    if tf is not None and hasattr(tf, "RaggedTensor") and isinstance(x, tf.RaggedTensor):
        try:
            return x.to_tensor().numpy()
        except Exception:
            pass
    try:
        return np.asarray(x)
    except Exception:
        return None
# ------------------------------------------


class FVLMWrapper(object):
    """
    F-VLM wrapper. Builds [1,91,1024] query tensor and calls SavedModel with keys:
      {'image': (1,Hm,Wm,3) bfloat16, 'text': (1,91,1024) float32}
    Returns (scores, boxes_cxcywh, labels0, times_dict).
    Boxes are normalized to the ORIGINAL PIL image.
    """
    def __init__(self,
                 repo_path,
                 model_name="resnet_50",
                 max_num_classes=91,
                 score_threshold=0.30,
                 fake_mode=False):
        t0 = time.perf_counter()

        self.repo_path = repo_path
        self.model_name = model_name
        self.max_num_classes = int(max_num_classes)
        self.score_threshold = float(score_threshold)
        self.fake_mode = bool(fake_mode)
        self.ready = False
        self._logged_model_keys = False

        if self.fake_mode:
            self.load_ms = 0.0
            rospy.logwarn("FVLMWrapper: FAKE mode enabled.")
            return

        try:
            import sys
            sys.path.insert(0, self.repo_path)
            from utils import clip_utils
            from demo_utils import input_utils
            import tensorflow as tf

            self.tf = tf
            self.clip_text_fn = clip_utils.get_clip_text_fn(self.model_name)
            self.parser_fn = input_utils.get_maskrcnn_parser()
            self.model_short = self.model_name.replace("resnet_", "r")

            # Background & empty embeddings
            embed_path = os.path.join(self.repo_path, "data", f"{self.model_short}_bg_empty_embed.npy")
            if not os.path.isfile(embed_path):
                raise FileNotFoundError(f"Missing embed file: {embed_path}")
            bg, empty = np.load(embed_path)
            self.background_embedding = bg[np.newaxis, :]
            self.empty_embedding      = empty[np.newaxis, :]

            # SavedModel
            ckpt_dir = os.path.join(self.repo_path, "checkpoints", self.model_short)
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"Missing SavedModel dir: {ckpt_dir}")
            self.saved_model = self.tf.saved_model.load(ckpt_dir)
            self._to_bf16 = lambda x: self.tf.cast(x, self.tf.bfloat16)

            self.ready = True
            self.load_ms = (time.perf_counter() - t0) * 1000.0
            rospy.loginfo("F-VLM loaded: model=%s  repo=%s  load_ms=%.1f",
                          self.model_name, self.repo_path, self.load_ms)
        except Exception as e:
            rospy.logerr("FVLMWrapper init failed: %s", e)
            self.fake_mode = True
            self.ready = False
            self.load_ms = (time.perf_counter() - t0) * 1000.0

    def _make_queries(self, texts, times):
        t0 = time.perf_counter()
        feats = [self.clip_text_fn(t) for t in texts] if texts else []
        text_emb = np.concatenate(feats, axis=0) if feats else np.zeros((0,1024), np.float32)
        out = [self.background_embedding, text_emb]
        need = self.max_num_classes - (1 + text_emb.shape[0])
        if need > 0:
            out.append(np.tile(self.empty_embedding, (need, 1)))
        text_91 = np.concatenate(out, axis=0)[:self.max_num_classes]
        times['text_ms'] = (time.perf_counter() - t0) * 1000.0
        return text_91[np.newaxis, ...].astype("float32")  # [1,91,1024]

    @staticmethod
    def _extract_embed_region(model_img):
        """
        Detect where the original image sits in the model frame by finding the
        non-zero region (parser pads with zeros). Returns (off_y, off_x, newH, newW).
        """
        mask = np.any(np.abs(model_img) > 1e-6, axis=2)  # True where content exists
        ys = np.where(mask.any(axis=1))[0]
        xs = np.where(mask.any(axis=0))[0]
        if ys.size == 0 or xs.size == 0:
            return 0, 0, model_img.shape[0], model_img.shape[1]
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        off_y, off_x = y0, x0
        newH = (y1 - y0 + 1)
        newW = (x1 - x0 + 1)
        return off_y, off_x, newH, newW

    def _prep_image(self, pil_image, times):
        t0 = time.perf_counter()
        np_image = np.array(pil_image)  # RGB (H0,W0,3)
        data = self.parser_fn({'image': np_image, 'source_id': np.array([0])})
        images = data['images']  # tf.Tensor [Hm,Wm,3]
        model_img = images.numpy() if hasattr(images, "numpy") else np.asarray(images)
        Hm, Wm = model_img.shape[0], model_img.shape[1]
        H0, W0 = np_image.shape[0], np_image.shape[1]
        times['prep_ms'] = (time.perf_counter() - t0) * 1000.0
        return model_img[np.newaxis, ...], (Hm, Wm), (H0, W0), model_img

    def detect(self, pil_image, text_list):
        # returns scores, boxes_cxcywh, labels, times_dict
        times = {}
        if self.fake_mode or not self.ready:
            w, h = pil_image.size
            s = float(min(w, h))
            bw = 0.4 * s / float(w); bh = 0.4 * s / float(h)
            times['total_ms'] = 0.1
            if not text_list:
                return (np.zeros((0,), np.float32),
                        np.zeros((0,4), np.float32),
                        np.zeros((0,), np.int64),
                        times)
            return (np.array([0.99], np.float32),
                    np.array([[0.5,0.5,bw,bh]], np.float32),
                    np.array([0], np.int64),
                    times)

        t_all0 = time.perf_counter()

        # text
        queries = self._make_queries(text_list, times)

        # image prep
        img_np, (Hm, Wm), (H0, W0), model_img = self._prep_image(pil_image, times)

        # inference
        t0 = time.perf_counter()
        tf = self.tf
        inputs = {
            'image': self._to_bf16(tf.convert_to_tensor(img_np)),
            'text' : tf.convert_to_tensor(queries, dtype=tf.float32),
        }
        output = self.saved_model(inputs)
        times['infer_ms'] = (time.perf_counter() - t0) * 1000.0

        # unpack & map back
        t0 = time.perf_counter()
        flat = _flatten_dict(output)
        if not self._logged_model_keys:
            try:
                rospy.loginfo("F-VLM output keys: %s", ", ".join(sorted(flat.keys())))
            except Exception:
                pass
            self._logged_model_keys = True

        boxes, _   = next(((v, k) for k, v in flat.items() if any(n in k.lower() for n in
                          ["detection_boxes","nmsed_boxes","boxes","detections/detection_boxes","detections/boxes"])), (None, None))
        scores, _  = next(((v, k) for k, v in flat.items() if any(n in k.lower() for n in
                          ["detection_scores","nmsed_scores","scores","detections/detection_scores","detections/scores"])), (None, None))
        classes, _ = next(((v, k) for k, v in flat.items() if any(n in k.lower() for n in
                          ["detection_classes","nmsed_classes","classes","labels","detections/detection_classes","detections/classes","detections/labels"])), (None, None))

        if boxes is None or scores is None or classes is None:
            times['post_ms'] = (time.perf_counter() - t0) * 1000.0
            times['total_ms'] = (time.perf_counter() - t_all0) * 1000.0
            return (np.zeros((0,), np.float32),
                    np.zeros((0,4), np.float32),
                    np.zeros((0,), np.int64),
                    times)

        boxes  = _to_numpy(boxes);  scores = _to_numpy(scores);  classes = _to_numpy(classes)
        if boxes.ndim  >= 2 and boxes.shape[0]  == 1: boxes  = boxes[0]
        if scores.ndim >= 2 and scores.shape[0] == 1: scores = scores[0]
        if classes.ndim>= 2 and classes.shape[0]== 1: classes= classes[0]
        classes = classes.astype(np.int64, copy=False)
        bn = boxes.astype(np.float32)

        # model coords in pixels
        if bn.max() <= 1.5:
            y1_m = bn[:,0]*Hm; x1_m = bn[:,1]*Wm
            y2_m = bn[:,2]*Hm; x2_m = bn[:,3]*Wm
        else:
            y1_m = bn[:,0];     x1_m = bn[:,1]
            y2_m = bn[:,2];     x2_m = bn[:,3]

        # exact invert using detected padding region
        off_y, off_x, newH, newW = self._extract_embed_region(model_img)
        x1_o = (x1_m - off_x) * (float(W0) / max(newW, 1e-6))
        y1_o = (y1_m - off_y) * (float(H0) / max(newH, 1e-6))
        x2_o = (x2_m - off_x) * (float(W0) / max(newW, 1e-6))
        y2_o = (y2_m - off_y) * (float(H0) / max(newH, 1e-6))

        # clip & normalize to ORIGINAL PIL
        x1_o = np.clip(x1_o, 0, W0 - 1); y1_o = np.clip(y1_o, 0, H0 - 1)
        x2_o = np.clip(x2_o, 0, W0 - 1); y2_o = np.clip(y2_o, 0, H0 - 1)

        x1 = x1_o / max(W0, 1); y1 = y1_o / max(H0, 1)
        x2 = x2_o / max(W0, 1); y2 = y2_o / max(H0, 1)
        cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
        bw = np.clip((x2 - x1), 1e-6, 1.0); bh = np.clip((y2 - y1), 1e-6, 1.0)
        boxes_cxcywh = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)

        # labels 0-based
        zero_based_ok = (classes.min() >= 0) and (classes.max() <= len(text_list) - 1)
        labels = classes if zero_based_ok else (classes - 1)
        valid = (labels >= 0) & (labels < len(text_list))

        times['post_ms'] = (time.perf_counter() - t0) * 1000.0
        times['total_ms'] = (time.perf_counter() - t_all0) * 1000.0

        if not np.any(valid):
            return (np.zeros((0,), np.float32),
                    np.zeros((0,4), np.float32),
                    np.zeros((0,), np.int64),
                    times)

        return (scores[valid].astype(np.float32, copy=False),
                boxes_cxcywh[valid],
                labels[valid].astype(np.int64, copy=False),
                times)


class FVLMServer(object):
    def __init__(self):
        rospy.init_node("fvlm_server")
        self.server_start_wall = time.time()

        # Params
        self.rgb_topic           = rospy.get_param("~rgb_topic", "/nico/vision/right")
        self.service_name        = rospy.get_param("~service_name", "object_detector")
        self.debug_topic         = rospy.get_param("~debug_img_topic", "/owlv2_server/result_image")

        self.score_threshold     = float(rospy.get_param("~score_threshold", 0.15))   # returned objects
        self.overlay_min_score   = float(rospy.get_param("~overlay_min_score", 0.05)) # drawn boxes
        self.keep_top_k          = int(rospy.get_param("~keep_top_k", rospy.get_param("~max_dets", 1)))
        self.fake_mode           = bool(rospy.get_param("~fake_mode", False))

        self.fvlm_repo_path      = rospy.get_param("~fvlm_repo_path", "")
        self.fvlm_model_name     = rospy.get_param("~fvlm_model", "resnet_50")
        self.max_num_classes     = int(rospy.get_param("~max_num_classes", 91))
        self.max_input_width     = int(rospy.get_param("~max_input_width", 0))

        self.save_debug_images   = bool(rospy.get_param("~save_debug_images", True))
        self.save_only_on_det    = bool(rospy.get_param("~save_only_on_detection", False))
        self.save_dir            = rospy.get_param("~save_dir", "/home/hb/elmira_captures")

        if self.save_debug_images:
            try: os.makedirs(self.save_dir, exist_ok=True)
            except Exception as e:
                rospy.logwarn("Could not create %s: %s", self.save_dir, e)
                self.save_debug_images = False

        self.bridge = cv_bridge.CvBridge()
        self.detector = FVLMWrapper(self.fvlm_repo_path, self.fvlm_model_name,
                                    self.max_num_classes, self.score_threshold, self.fake_mode)

        self.last_img_msg = None
        self.image_sub = rospy.Subscriber(self.rgb_topic, sensor_msgs.msg.Image,
                                          self._img_cb, queue_size=1, buff_size=6*(1024**2))
        self.srv = rospy.Service(self.service_name, DetectObjects, self.handle_detect)
        self.debug_pub = rospy.Publisher(self.debug_topic, sensor_msgs.msg.Image, latch=True, queue_size=1)
        self.timing_pub = rospy.Publisher("~timings", String, latch=True, queue_size=1)

        rospy.loginfo("FVLM server started: /%s  rgb=%s  keep_top_k=%d  load_ms=%.1f",
                      self.service_name, self.rgb_topic, self.keep_top_k, self.detector.load_ms)
        rospy.spin()

    def _img_cb(self, msg):
        self.last_img_msg = msg

    def handle_detect(self, req):
        req_t0 = time.perf_counter()
        wait_img_ms = 0.0

        # Grab frame
        img_msg = self.last_img_msg
        if img_msg is None:
            t0 = time.perf_counter()
            img_msg = rospy.wait_for_message(self.rgb_topic, sensor_msgs.msg.Image, timeout=1.0)
            wait_img_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        grab_ms = (time.perf_counter() - t0) * 1000.0

        # Optional downscale
        t0 = time.perf_counter()
        det_img = cv_img
        if self.max_input_width and cv_img.shape[1] > self.max_input_width:
            new_w = self.max_input_width
            new_h = int(cv_img.shape[0] * (float(new_w) / cv_img.shape[1]))
            det_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_det = Image.fromarray(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        preproc_ms = (time.perf_counter() - t0) * 1000.0

        # Detect
        scores, boxes_cxcywh, labels, times_det = self.detector.detect(pil_det, req.texts)

        # Draw overlay (boxes only; NO timing text)
        H, W = cv_img.shape[:2]
        vis = cv_img.copy()
        out_objs = []
        for i in range(len(labels)):
            s = float(scores[i])
            if s < self.overlay_min_score:
                continue
            cx, cy, bw, bh = boxes_cxcywh[i].tolist()
            x1 = int((cx - bw/2.0) * W); y1 = int((cy - bh/2.0) * H)
            x2 = int((cx + bw/2.0) * W); y2 = int((cy + bh/2.0) * H)
            x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
            x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
            label_idx = int(labels[i])
            label_text = req.texts[label_idx] if 0 <= label_idx < len(req.texts) else f"class_{label_idx}"
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis, f"{label_text} {s:.2f}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        # Return objects (threshold & top-k)
        if len(scores) > 0:
            order = np.argsort(-scores)
            for i in order:
                s = float(scores[i])
                if s < self.score_threshold:
                    continue
                label_idx = int(labels[i])
                cx, cy, bw, bh = boxes_cxcywh[i].tolist()
                label_text = req.texts[label_idx] if 0 <= label_idx < len(req.texts) else f"class_{label_idx}"
                out_objs.append(DetectedObject(label_text, s, float(cx), float(cy), float(bw), float(bh)))
                if 0 < self.keep_top_k <= len(out_objs):
                    break

        # Timings (terminal only)
        total_req_ms = (time.perf_counter() - req_t0) * 1000.0
        uptime_s = time.time() - self.server_start_wall
        summary = {
            "uptime_s": round(uptime_s, 3),
            "model_load_ms": round(self.detector.load_ms, 1),
            "wait_for_image_ms": round(wait_img_ms, 1),
            "grab_ms": round(grab_ms, 1),
            "preproc_ms": round(preproc_ms, 1),
            "text_ms": round(times_det.get("text_ms", 0.0), 1),
            "prep_ms(detector)": round(times_det.get("prep_ms", 0.0), 1),
            "infer_ms": round(times_det.get("infer_ms", 0.0), 1),
            "post_ms": round(times_det.get("post_ms", 0.0), 1),
            "detector_total_ms": round(times_det.get("total_ms", 0.0), 1),
            "request_total_ms": round(total_req_ms, 1),
            "returned_objects": len(out_objs),
        }
        rospy.loginfo("TIMINGS %s", json.dumps(summary, sort_keys=True))
        try:
            self.timing_pub.publish(String(data=json.dumps(summary)))
        except Exception:
            pass

        # Publish/save overlay image (no timing text)
        try:
            dbg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            dbg.header.stamp = rospy.Time.now()
            self.debug_pub.publish(dbg)
        except Exception as e:
            rospy.logwarn("Debug image publish failed: %s", e)

        if self.save_debug_images and (len(out_objs) > 0 or not self.save_only_on_det):
            try:
                ts = time.strftime("%Y%m%d_%H%M%S")
                raw_path = os.path.join(self.save_dir, f"raw_{ts}.jpg")
                ovl_path = os.path.join(self.save_dir, f"detect_{ts}.jpg")
                cv2.imwrite(raw_path, cv_img)
                cv2.imwrite(ovl_path, vis)
                rospy.loginfo("Saved images:\n  %s\n  %s", raw_path, ovl_path)
            except Exception as e:
                rospy.logwarn("Failed to save debug images: %s", e)

        return DetectObjectsResponse(out_objs)


if __name__ == "__main__":
    FVLMServer()

