#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import rospy
from std_msgs.msg import String
from elmira.srv import PromptTextLLM, PromptTextLLMResponse
from elmira.srv import PromptVisionLLM, PromptVisionLLMResponse
from elmira.srv import CheckLLMObjectVisibility, CheckLLMObjectVisibilityResponse
from elmira.srv import DetectObjects  # service used by your object detector (now FVLM)

HOTWORD = rospy.get_param("/rule_llm_server/hotword", "nico")
ACTIONS = ["touch","point","show","push","push_left","push_right","move","go","goto","grab","pick","place","look"]

# Simple patterns: "<verb> <object>", "please <verb> the <object>", "move to <object>"
PATTERNS = [
    r"^(?P<verb>" + "|".join(ACTIONS) + r")\s+(the\s+)?(?P<object>.+)$",
    r"^please\s+(?P<verb>" + "|".join(ACTIONS) + r")\s+(the\s+)?(?P<object>.+)$",
    r"^(move|go|goto)\s+(to\s+)?(?P<object>.+)$",
    r"^can you\s+(?P<verb>" + "|".join(ACTIONS) + r")\s+(the\s+)?(?P<object>.+)\??$",
]

QUIT_WORDS = ["quit","exit","goodbye","bye","stop session","end"]
DESCRIBE_WORDS = ["describe","what do you see","what can you see","look around","list objects"]

def strip_hotword(text):
    t = text.strip()
    if HOTWORD and t.lower().startswith(HOTWORD + ","):
        t = t[len(HOTWORD)+1:].strip()
    return t

def parse_intent(text):
    t = strip_hotword(text.lower())
    # quit?
    if any(q == t or t.startswith(q) for q in QUIT_WORDS):
        return {"actions": [{"action":"speak","text":"Goodbye!"},{"action":"quit"}]}
    # describe?
    if any(kw in t for kw in DESCRIBE_WORDS):
        return {"actions": [{"action":"speak","text":"Let me take a look."},{"action":"describe"}]}

    # verb + object
    for p in PATTERNS:
        m = re.match(p, t)
        if m:
            verb = (m.groupdict().get("verb") or "point").lower()
            obj  = m.groupdict().get("object","").strip()
            # normalize verbs
            verb_map = {
                "show":"show",
                "point":"show",
                "touch":"touch",
                "push":"push",
                "push_left":"push_left",
                "push_right":"push_right",
                "move":"move",
                "go":"move",
                "goto":"move",
                "grab":"grab",
                "pick":"pick",
                "place":"place",
                "look":"show",
            }
            atype = verb_map.get(verb, "show")
            # Build the minimal action list your SMACH expects
            return {"actions":[
                {"action":"speak","text":f"Okay, {atype} the {obj}."},
                {"action":"act","object":obj,"type":atype}
            ]}
    # fallback: ask to rephrase
    return {"actions":[{"action":"speak","text":"I didn’t catch the object. Try: 'touch the banana' or 'point red cup'."}]}

class RuleLLMServer(object):
    def __init__(self):
        rospy.init_node("rule_llm_server")

        # Connect to the object detector (your FVLM server). Same service name for drop-in:
        self.objdet_name = rospy.get_param("~object_detector_service", "object_detector")
        rospy.wait_for_service(self.objdet_name)
        self.detect = rospy.ServiceProxy(self.objdet_name, DetectObjects)

        # Set up services with the SAME names as your GPT4Server
        self.srv_chat = rospy.Service("llm_chat", PromptTextLLM, self.handle_chat)
        self.srv_vis  = rospy.Service("llm_vision", PromptVisionLLM, self.handle_vision)
        self.srv_objv = rospy.Service("llm_object_visibility", CheckLLMObjectVisibility, self.handle_visibility)

        rospy.loginfo("RuleLLMServer up. Using object detector service: /%s", self.objdet_name)
        rospy.spin()

    # Return a JSON string like GPT4Server did
    def handle_chat(self, req):
        plan = parse_intent(req.prompt)
        return json.dumps(plan, ensure_ascii=False)

    # Keep vision prompt simple (you can extend later)
    def handle_vision(self, req):
        # For now, trigger a describe pass so your pipeline captures an image
        plan = {"actions":[{"action":"speak","text":"Capturing a view."},{"action":"describe"}]}
        return json.dumps(plan, ensure_ascii=False)

    # Use FVLM to check if object is visible
    def handle_visibility(self, req):
        # Heuristic: extract last noun-ish token if you pass full phrases; or just use the whole prompt
        text = req.prompt.strip()
        # Call the detector: it expects a list of category strings
        try:
            resp = self.detect([text])  # DetectObjects.srv: texts: string[]
            seen = len(resp.objects) > 0
            if seen:
                return CheckLLMObjectVisibilityResponse(True, "")
            else:
                return CheckLLMObjectVisibilityResponse(False, "SYSTEM: I can’t see it yet. Please place it in view.")
        except Exception as e:
            rospy.logwarn("Visibility check failed: %s", str(e))
            return CheckLLMObjectVisibilityResponse(False, "SYSTEM: Vision not available right now.")

if __name__ == "__main__":
    RuleLLMServer()

