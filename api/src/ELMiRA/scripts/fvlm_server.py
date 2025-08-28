#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, numpy as np, cv2
from PIL import Image
import rospy, cv_bridge, sensor_msgs.msg
from elmira.msg import DetectedObject
from elmira.srv import DetectObjects, DetectObjectsResponse

# ---------- small utils ----------
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
    if isinstance(x, np.ndarray): return x
    if tf is not None and hasattr(tf, "Tensor") and isinstance(x, tf.Tensor):
        try: return x.numpy()
        except Exception: pass
    if tf is not None and hasattr(tf, "RaggedTensor") and isinstance(x, tf.RaggedTensor):
        try: return x.to_tensor().numpy()
        except Exception: pass
    try: return np.asarray(x)
    except Exception: return None
# ----------------------------------

class FVLMWrapper(object):
    """
    Runs F-VLM and returns:
      scores: (N,) float32
      boxes : (N,4) float32 in NORMALIZED cx,cy,w,h w.r.t. the **ORIGINAL PIL** image
      labels: (N,) int64  (0-based index into the provided text list)
    """
    def __init__(self, repo_path, model_name="resnet_50", max_num_classes=91,
                 score_threshold=0.30, fake_mode=False):
        self.repo_path = repo_path
        self.model_name = model_name
        self.max_num_classes = int(max_num_classes)
        self.score_threshold = float(score_threshold)
        self.fake_mode = bool(fake_mode)
        self.ready = False
        self._logged_model_keys = False

        if self.fake_mode:
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

            # background/empty embeddings
            embed_path = os.path.join(self.repo_path, "data", f"{self.model_short}_bg_empty_embed.npy")
            if not os.path.isfile(embed_path):
                raise FileNotFoundError(f"Missing embed file: {embed_path}")
            bg, empty = np.load(embed_path)
            self.background_embedding = bg[np.newaxis, :]
            self.empty_embedding      = empty[np.newaxis, :]

            ckpt_dir = os.path.join(self.repo_path, "checkpoints", self.model_short)
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"Missing SavedModel dir: {ckpt_dir}")
            self.saved_model = self.tf.saved_model.load(ckpt_dir)
            self._to_bf16 = lambda x: self.tf.cast(x, self.tf.bfloat16)

            self.ready = True
            rospy.loginfo("F-VLM loaded: model=%s  repo=%s", self.model_name, self.repo_path)
        except Exception as e:
            rospy.logerr("FVLMWrapper init failed: %s", e)
            self.fake_mode = True

    def _make_queries(self, texts):
        feats = [self.clip_text_fn(t) for t in texts] if texts else []
        text_emb = np.concatenate(feats, axis=0) if feats else np.zeros((0,1024), np.float32)
        out = [self.background_embedding, text_emb]
        need = self.max_num_classes - (1 + text_emb.shape[0])
        if need > 0: out.append(np.tile(self.empty_embedding, (need, 1)))
        text_91 = np.concatenate(out, axis=0)[:self.max_num_classes]
        return text_91[np.newaxis, ...].astype("float32")  # [1,91,1024]

    @staticmethod
    def _extract_embed_region(model_img):
        """
        Given model frame [Hm,Wm,3] (as float), find where the original image sits by
        detecting zero-padding. Returns (off_y, off_x, newH, newW). If no pad found,
        returns (0,0,Hm,Wm).
        """
        # model_img is float; padding is zeros
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

    def _prep_image(self, pil_image):
        np_image = np.array(pil_image)  # RGB (H0,W0,3)
        data = self.parser_fn({'image': np_image, 'source_id': np.array([0])})
        images = data['images']  # tf.Tensor [Hm,Wm,3]
        model_img = images.numpy() if hasattr(images, "numpy") else np.asarray(images)
        Hm, Wm = model_img.shape[0], model_img.shape[1]
        H0, W0 = np_image.shape[0], np_image.shape[1]
        return model_img[np.newaxis, ...], (Hm, Wm), (H0, W0), model_img  # include raw frame for pad-detect

    def detect(self, pil_image, text_list):
        if self.fake_mode or not self.ready:
            w, h = pil_image.size
            s = float(min(w, h))
            bw = 0.4 * s / float(w); bh = 0.4 * s / float(h)
            if not text_list:
                return np.zeros((0,), np.float32), np.zeros((0,4), np.float32), np.zeros((0,), np.int64)
            return np.array([0.99], np.float32), np.array([[0.5,0.5,bw,bh]], np.float32), np.array([0], np.int64)

        tf = self.tf
        queries = self._make_queries(text_list)
        img_np, (Hm, Wm), (H0, W0), model_img = self._prep_image(pil_image)

        inputs = {
            'image': self._to_bf16(tf.convert_to_tensor(img_np)),
            'text' : tf.convert_to_tensor(queries, dtype=tf.float32),
        }
        output = self.saved_model(inputs)
        flat = _flatten_dict(output)
        if not self._logged_model_keys:
            try: rospy.loginfo("F-VLM output keys: %s", ", ".join(sorted(flat.keys())))
            except Exception: pass
            self._logged_model_keys = True

        boxes, _   = next(((v, k) for k, v in flat.items() if any(n in k.lower() for n in
                          ["detection_boxes","nmsed_boxes","boxes","detections/detection_boxes","detections/boxes"])), (None, None))
        scores, _  = next(((v, k) for k, v in flat.items() if any(n in k.lower() for n in
                          ["detection_scores","nmsed_scores","scores","detections/detection_scores","detections/scores"])), (None, None))
        classes, _ = next(((v, k) for k, v in flat.items() if any(n in k.lower() for n in
                          ["detection_classes","nmsed_classes","classes","labels","detections/detection_classes","detections/classes","detections/labels"])), (None, None))
        if boxes is None or scores is None or classes is None:
            return np.zeros((0,), np.float32), np.zeros((0,4), np.float32), np.zeros((0,), np.int64)

        # squeeze batch
        boxes  = _to_numpy(boxes);  scores = _to_numpy(scores);  classes = _to_numpy(classes)
        if boxes.ndim  >= 2 and boxes.shape[0]  == 1: boxes  = boxes[0]
        if scores.ndim >= 2 and scores.shape[0] == 1: scores = scores[0]
        if classes.ndim>= 2 and classes.shape[0]== 1: classes= classes[0]
        classes = classes.astype(np.int64, copy=False)
        bn = boxes.astype(np.float32)

        # model coords (pixels)
        if bn.max() <= 1.5:
            y1_m = bn[:,0]*Hm; x1_m = bn[:,1]*Wm
            y2_m = bn[:,2]*Hm; x2_m = bn[:,3]*Wm
        else:
            y1_m = bn[:,0];     x1_m = bn[:,1]
            y2_m = bn[:,2];     x2_m = bn[:,3]

        # ---- infer exact padding by looking at zero borders ----
        off_y, off_x, newH, newW = self._extract_embed_region(model_img)
        # invert embedding to original pixels
        x1_o = (x1_m - off_x) * (float(H0)/newH) * (newW/float(Wm)) * (float(W0)/float(W0))  # keep formula symmetric
        y1_o = (y1_m - off_y) * (float(W0)/newW) * (newH/float(Hm)) * (float(H0)/float(H0))
        # actually the simple invert is:
        x1_o = (x1_m - off_x) * (float(W0) / max(newW, 1e-6))
        y1_o = (y1_m - off_y) * (float(H0) / max(newH, 1e-6))
        x2_o = (x2_m - off_x) * (float(W0) / max(newW, 1e-6))
        y2_o = (y2_m - off_y) * (float(H0) / max(newH, 1e-6))

        # clip & normalize
        x1_o = np.clip(x1_o, 0, W0-1); y1_o = np.clip(y1_o, 0, H0-1)
        x2_o = np.clip(x2_o, 0, W0-1); y2_o = np.clip(y2_o, 0, H0-1)
        x1 = x1_o / max(W0,1); y1 = y1_o / max(H0,1)
        x2 = x2_o / max(W0,1); y2 = y2_o / max(H0,1)
        cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
        bw = np.clip((x2-x1), 1e-6, 1.0); bh = np.clip((y2-y1), 1e-6, 1.0)
        boxes_cxcywh = np.stack([cx,cy,bw,bh], axis=1).astype(np.float32)

        # 0-based labels
        zero_based_ok = (classes.min() >= 0) and (classes.max() <= len(text_list)-1)
        labels = classes if zero_based_ok else (classes-1)
        valid = (labels >= 0) & (labels < len(text_list))
        if not np.any(valid):
            return np.zeros((0,),np.float32), np.zeros((0,4),np.float32), np.zeros((0,),np.int64)

        return scores[valid].astype(np.float32, copy=False), boxes_cxcywh[valid], labels[valid].astype(np.int64, copy=False)


class FVLMServer(object):
    def __init__(self):
        rospy.init_node("fvlm_server")
        # Topics & params
        self.rgb_topic         = rospy.get_param("~rgb_topic", "/nico/vision/right")
        self.service_name      = rospy.get_param("~service_name", "object_detector")
        self.debug_topic       = rospy.get_param("~debug_img_topic", "/owlv2_server/result_image")
        self.score_threshold   = float(rospy.get_param("~score_threshold", 0.15))   # returns
        self.overlay_min_score = float(rospy.get_param("~overlay_min_score", 0.05))# draws
        self.keep_top_k        = int(rospy.get_param("~keep_top_k", rospy.get_param("~max_dets", 1)))
        self.fake_mode         = bool(rospy.get_param("~fake_mode", False))
        self.fvlm_repo_path    = rospy.get_param("~fvlm_repo_path", "")
        self.fvlm_model_name   = rospy.get_param("~fvlm_model", "resnet_50")
        self.max_num_classes   = int(rospy.get_param("~max_num_classes", 91))
        self.max_input_width   = int(rospy.get_param("~max_input_width", 0))

        self.bridge = cv_bridge.CvBridge()
        self.detector = FVLMWrapper(self.fvlm_repo_path, self.fvlm_model_name,
                                    self.max_num_classes, self.score_threshold, self.fake_mode)

        self.last_img_msg = None
        self.image_sub = rospy.Subscriber(self.rgb_topic, sensor_msgs.msg.Image, self._img_cb,
                                          queue_size=1, buff_size=6*(1024**2))
        self.srv = rospy.Service(self.service_name, DetectObjects, self.handle_detect)
        self.debug_pub = rospy.Publisher(self.debug_topic, sensor_msgs.msg.Image, latch=True, queue_size=1)
        rospy.loginfo("FVLM server started: /%s  rgb=%s  keep_top_k=%d",
                      self.service_name, self.rgb_topic, self.keep_top_k)
        rospy.spin()

    def _img_cb(self, msg): self.last_img_msg = msg

    def handle_detect(self, req):
        # frame
        img_msg = self.last_img_msg or rospy.wait_for_message(self.rgb_topic, sensor_msgs.msg.Image, timeout=1.0)
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # optional downscale for detector (keeps aspect)
        det_img = cv_img
        if self.max_input_width and cv_img.shape[1] > self.max_input_width:
            new_w = self.max_input_width
            new_h = int(cv_img.shape[0] * (float(new_w) / cv_img.shape[1]))
            det_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # run detector
        pil_det = Image.fromarray(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        scores, boxes_cxcywh, labels = self.detector.detect(pil_det, req.texts)

        H, W = cv_img.shape[:2]
        vis = cv_img.copy()
        out_objs = []

        # draw
        for i in range(len(labels)):
            s = float(scores[i])
            if s < self.overlay_min_score: continue
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

        # return objects (top-k by score)
        if len(scores) > 0:
            order = np.argsort(-scores)
            for i in order:
                s = float(scores[i])
                if s < self.score_threshold: continue
                label_idx = int(labels[i])
                cx, cy, bw, bh = boxes_cxcywh[i].tolist()
                label_text = req.texts[label_idx] if 0 <= label_idx < len(req.texts) else f"class_{label_idx}"
                out_objs.append(DetectedObject(label_text, s, float(cx), float(cy), float(bw), float(bh)))
                if 0 < self.keep_top_k <= len(out_objs): break

        # publish debug
        try:
            dbg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            dbg.header.stamp = rospy.Time.now()
            self.debug_pub.publish(dbg)
        except Exception as e:
            rospy.logwarn("Debug image publish failed: %s", e)

        # save images
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join("/home/hb/elmira_captures", f"raw_{ts}.jpg")
            ovl_path = os.path.join("/home/hb/elmira_captures", f"detect_{ts}.jpg")
            os.makedirs("/home/hb/elmira_captures", exist_ok=True)
            cv2.imwrite(raw_path, cv_img)
            cv2.imwrite(ovl_path, vis)
            rospy.loginfo("Saved images:\n  %s\n  %s", raw_path, ovl_path)
        except Exception as e:
            rospy.logwarn("Failed to save debug images: %s", e)

        return DetectObjectsResponse(out_objs)

if __name__ == "__main__":
    FVLMServer()

