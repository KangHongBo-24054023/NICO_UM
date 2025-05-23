#!/usr/bin/env python3

import json
import os
import openai
from openai import OpenAI
import base64
import cv2
import cv_bridge
import rospy
import sensor_msgs.msg
from io import BytesIO, BufferedReader

from elmira.srv import PromptTextLLM, PromptVisionLLM, CheckLLMObjectVisibility, PromptTextLLMResponse


class GPT4Server:

    def __init__(self):
        rospy.init_node("gpt4_server")

        self.bridge = cv_bridge.CvBridge()

        if "OPENAI_API_KEY" not in os.environ:
            rospy.logerr("OPENAI_API_KEY is not set")
            exit(1)

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.assistant = self.create_assistant()  # FIXME reset context?
        self.thread = self.client.beta.threads.create()
        self.files = []
        # TODO merge chat and vision?
        rospy.Service("llm_chat", PromptTextLLM, self.text_prompt_request_handler)
        rospy.Service(
            "llm_object_visibility",
            CheckLLMObjectVisibility,
            self.visibility_request_handler,
        )
        rospy.Service(
            "llm_vision",
            PromptVisionLLM,
            self.vision_prompt_request_handler,
        )
        rospy.loginfo("GPT4 started successfully")
        rospy.spin()
        # remove all images from this session
        for file in self.files:
            self.client.files.delete(file)

    def create_assistant(self):
        name = "NICO"
        instructions = (
            "You are a child-size humanoid robot, named NICO, that interacts with a human user. "
            + "Your task is to manipulate objects placed on a table at which you are seated, as well as communicating with the human. "
            + "You will either receive a 'USER:' query with a transcription of the user's verbal input or a 'SYSTEM:' query whenever one of your systems returns feedback or status messages."
            + "You need to respond with a list of actions to trigger your different systems to interact with the user. "
            + "These actions will be processed and fully executed in sequence, before the user is prompted for input again or the interaction ends. "
            + "Speak: In order to verbally respond to the user, you should add a 'speak' action to the list of actions with an additional 'text' field. The text will be used to produce speech with your TTS module. "
            + "Describe: Whenever you need visual information to respond to a user query about your surroundings or objects on the table, you need to actively request it by adding a 'describe' action to the list of actions. This lets you look at the table and take an image with the right eye camera which you will receive as input. "
            + "Act: When instructed by the user to interact with objects on the table, you have to add the 'act' action, triggering your object detection and IK solver to produce physical actions with the left or right arm. "
            + "You need to add a key for the 'type' of action and the target 'object' specified by the user for your systems to know which action to choose. "
            + "Valid types are: 'touch' to touch the object with your hand, 'push' to move ithe object forward, 'push_left' to move it to the left, 'push_right' to move it to the right and 'show' to point towards it. "
            + "Quit: To end the interaction entirely, you should output the 'quit' signal with no additional parameters. "
            + "Try to be responsive and transparent to the user by adding 'speak' actions before or after executing other actions where appropriate. "
            + "Please always output your response as a valid, single line json containing the list of actions, for example "
            + "`{'actions': [{'action': 'speak', 'text': 'Sure, I can do that for you.'}, {'action': 'act', 'object': 'banana', 'type': 'touch'}]}`, "
            + "`{'actions': [[{'action': 'describe'}]}` or "
            + "`{'actions': [[{'action': 'speak', 'text': 'Goodbye! I hope we see each other again.'}, {'action': 'quit'}]}`."
        )
        tools = []  # TODO: use function calling?
        model = (
            "gpt-4o"  # "gpt-4-turbo-preview",  # TODO: use gpt-4o, add image directly
        )
        response_format = {"type": "json_object"}

        assistant_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".assistant_id"
        )
        if os.path.isfile(assistant_file):
            with open(assistant_file, "r") as f:
                assistant_id = f.readline()
            assistant = self.client.beta.assistants.update(
                assistant_id,
                name=name,
                instructions=instructions,
                tools=tools,
                model=model,
                response_format=response_format,
            )
        else:
            assistant = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=tools,
                model=model,
                response_format=response_format,
            )
            with open(assistant_file, "w") as f:
                f.write(assistant.id)
        return assistant

    def create_vision_object_exists(self, prompt, base64_image):
        response = self.client.chat.completions.create(
            model="gpt-4o",  # "gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a visual verification system of a child-size humanoid robot, named NICO, that is tasked by a human user to manipulate objects placed on the table in front of it. "
                    + "Based on the given image, is the target object in the given user prompt visible on the table? If yes, output `{'object_visible': true}` as json. If not, output `{'object_visible': false}` and add a short, concise `'system_message':` which the main chat system of the robot can use to formulate its response to the user.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=100,
            temperature=0.8,
            n=1,
            response_format={"type": "json_object"},
        )
        return response

    def text_prompt_request_handler(self, request):
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id, role="user", content=request.prompt
        )
        # client.beta.threads.messages.list(thread_id=thread.id).data
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id, assistant_id=self.assistant.id
        )
        r = rospy.Rate(1)  # 1 Hz
        while run.status in ["queued", "in_progress", "cancelling"]:
            r.sleep()  # Wait for 1 second
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id, run_id=run.id
            )

        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            rospy.loginfo(f"Messages returned:\n{messages.data}")
            try:
                json_string = messages.data[0].content[0].text.value
                if not json_string.strip():
                    rospy.logwarn("LLM response is empty!")
                return PromptTextLLMResponse(response = json_string)
            except Execption as e:
                rospy.logerr(f"Eroor extracting LLM response:{e}")
                return PromptTextLLMResponse(response="")
        else:
            rospy.logerr(run.status)
            return PromptTextLLMResponse(response="")

    def vision_prompt_request_handler(self, request):
        img_msg = rospy.wait_for_message(
            "/nico/vision/right",
            sensor_msgs.msg.Image,
        )
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5)

        # Upload image data to openai directly without saving inbetween
        _, buffer = cv2.imencode(".png", cv_image)
        image_stream = BytesIO(buffer)
        image_stream.name = "vision_right_eye.png"
        file = self.client.files.create(
            file=BufferedReader(image_stream), purpose="vision"
        )
        # store file id to delete at shutdown
        self.files.append(file.id)
        # send image message to assistant
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=[{"type": "image_file", "image_file": {"file_id": file.id}}],
        )
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id, assistant_id=self.assistant.id
        )
        # wait for response
        r = rospy.Rate(1)  # 1 Hz
        while run.status in ["queued", "in_progress", "cancelling"]:
            r.sleep()  # Wait for 1 second
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id, run_id=run.id
            )
        # return llm response
        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            json_string = messages.data[0].content[0].text.value
            return json_string
        else:
            rospy.logerr(run.status)

    def visibility_request_handler(self, request):
        img_msg = rospy.wait_for_message(
            "/nico/vision/right",
            sensor_msgs.msg.Image,
        )
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5)

        # Getting the base64 string
        _, buffer = cv2.imencode(".png", cv_image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        response = self.create_vision_object_exists(request.prompt, base64_image)
        response = json.loads(response.choices[0].message.content.strip())
        if response["object_visible"]:
            return True, ""
        else:
            return False, f"SYSTEM: {response['system_message']}"


# Run main function
if __name__ == "__main__":
    try:
        GPT4Server()
    except openai.OpenAIError as e:
        rospy.logerr(e)
# EOF
