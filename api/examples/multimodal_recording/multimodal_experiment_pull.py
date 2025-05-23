# MUltimodal Recording

# Erik Strahl
# Matthias Kerezl

# GNU GPL License


from nicomotion import Motion
from nicomotion import Mover
import pypot.dynamixel
from time import sleep
import datetime
from nicotouch.optoforcesensors import optoforce

import logging
from nicovision import ImageRecorder
import os
from os.path import dirname, abspath
from time import sleep
import datetime
import sys
import cv2

from nicoaudio import pulse_audio_recorder

from subprocess import call

fnl="left_cam_synced_data.csv"
fnr="right_cam_synced_data.csv"
robot=None

import sqlite3
import random

from subprocess import call

import shutil

#experiment definitions (Objects and number of graspings)
#definition of objects
#objects =["blue ball","blue_plush ball","red_plush ball", "orange_plush ball", \
#          "white cube", "big_yellow die", "medium_yellow die","small_yellow die", \
#          "yellow sponge", "green sponge","blue tissues","pink tissues", \
#          "yellow apple", "light_green apple","heavy_green apple", "realistic_green apple", \
#          "red car","blue car","yellow car", "green car", \
#          "light tomato", "heavy tomato","small_red ball", "large_red ball", \
#          "small banana", "plush banana", "heavy banana","aubergine banana", \
#          "yellow duck","purple duck","orange fish","yellow seal"]

#Test objects for Marcel
objects =["blue ball","blue_plush ball","red_plush ball", "orange_plush ball", \
          "big_yellow die","small_yellow die", "yellow car", "green apple", \
          "light tomato", "heavy tomato","red ball", "red apple"]

#action
action="pull"

#data_directory
data_directory="/data2/20180608_multimodal_recording_pilot_data"

#definition for numbers per object
number_of_samples_per_object=10

#definition of Maximum current - This protects the hands from breaking!! Do not change this, if you do not know!
MAX_CUR_FINGER=120
MAX_CUR_THUMB=100

# Data for SQlite Access
# Experiment Data will be stored in two table database
# with samples for every sample of a grasped onject and subsample for every motor step (like for 10 degrees)

print "database file: " + data_directory+"/multimodal_experiment.db"

connection = sqlite3.connect(data_directory+"/multimodal_experiment.db")



cursor = connection.cursor()
# status: 0-succesful , 1- overload, 2 - former_overload


format_str_sample = """INSERT INTO sample (sample_number,object_name , action, timecode)
    VALUES (NULL, "{object_name}", "{action}","{timecode}");"""


# Data for robot movement in high level mode
# move joints with 10 per cent of the speed and hand joints with full speed
fMS = 0.01
fMS_hand=1.0


#Pandas structures for storing joint data
import pandas as pd
columns = ["r_shoulder_z_pos","r_shoulder_y_pos","r_arm_x_pos","r_elbow_y_pos","r_wrist_z_pos","r_wrist_x_pos","r_indexfingers_x_pos","r_thumb_x_pos","head_z_pos", "head_y_pos", \
           "r_shoulder_z_cur","r_shoulder_y_cur","r_arm_x_cur","r_elbow_y_cur","r_wrist_z_cur","r_wrist_x_cur","r_indexfingers_x_cur","r_thumb_x_cur","head_z_cur", "head_y_cur", \
           "isotime","touch_isotime","touch_count","touch_x","touch_y","touch_z"]
dfl = pd.DataFrame(columns=columns)
dfr = pd.DataFrame(columns=columns)


def write_joint_data(robot,df,iso_time):
	
	#df = pd.DataFrame.append(data={"r_arm_x":[robot.getAngle("r_arm_x")],"r_elbow_y":[robot.getAngle("r_elbow_y")],"head_z":[robot.getAngle("head_z")],"isotime":[iso_time]})
	(stime, counter, status, x, y, z, checksum)=optoforce_sensor.get_sensor_all()
	dfn = pd.DataFrame(data={"r_shoulder_z_pos":[robot.getAngle("r_shoulder_z")],  \
	                         "r_shoulder_y_pos":[robot.getAngle("r_shoulder_y")],  \
	                         "r_arm_x_pos":[robot.getAngle("r_arm_x")],  \
	                         "r_elbow_y_pos":[robot.getAngle("r_elbow_y")],  \
	                         "r_wrist_z_pos":[robot.getAngle("r_wrist_z")],  \
	                         "r_wrist_x_pos":[robot.getAngle("r_wrist_x")],  \
	                         "r_indexfingers_x_pos":[robot.getAngle("r_indexfingers_x")],  \
	                         "r_thumb_x_pos":[robot.getAngle("r_thumb_x")],  \
	                         "head_z_pos":[robot.getAngle("head_z")],  \
	                         "head_y_pos":[robot.getAngle("head_z")],  \
	                         "r_shoulder_z_cur":[robot.getCurrent("r_shoulder_z")],  \
	                         "r_shoulder_y_cur":[robot.getCurrent("r_shoulder_y")],  \
	                         "r_arm_x_cur":[robot.getCurrent("r_arm_x")],  \
	                         "r_elbow_y_cur":[robot.getCurrent("r_elbow_y")],  \
	                         "r_wrist_z_cur":[robot.getCurrent("r_wrist_z")],  \
	                         "r_wrist_x_cur":[robot.getCurrent("r_wrist_x")],  \
	                         "r_indexfingers_x_cur":[robot.getCurrent("r_indexfingers_x")],  \
	                         "r_thumb_x_cur":[robot.getCurrent("r_thumb_x")],  \
	                         "head_z_cur":[robot.getCurrent("head_z")],  \
	                         "head_y_cur":[robot.getCurrent("head_z")],  \
                             "isotime":[iso_time], \
                             "touch_isotime":[stime], \
                             "touch_count":[counter], \
                             "touch_x":[x], \
                             "touch_y":[y], \
                             "touch_z":[z]})
	df = pd.concat([df, dfn], ignore_index=True)
	
	#df = pd.DataFrame(data={"r_arm_x":[robot.getAngle("r_arm_x")],"r_elbow_y":[robot.getAngle("r_elbow_y")],"head_z":[robot.getAngle("head_z")]})
	return(df)



class leftcam_ImageRecorder(ImageRecorder.ImageRecorder):
	
	def custom_callback(self, iso_time,frame):
		
		global dfl;
		
		#write joint data
		
		dfl=write_joint_data(robot,dfl,iso_time)
		
		#small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
		#small=cv2.flip(small,-1)
		#return(small)
		return(frame) 

class rightcam_ImageRecorder(ImageRecorder.ImageRecorder):
	
	def custom_callback(self, iso_time,frame):
		
		global dfr;
		
		#write joint data
		
		dfr=write_joint_data(robot,dfr,iso_time)
		
		#small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)		
		#return(small)
		return(frame) 



def get_sampled_numbers_for_object(object_name):

    sql="SELECT * FROM sample where object_name='" + object_name + "' and action='"+action+"';"
    cursor.execute(sql)
    result = cursor.fetchall()
    return len(result)

def get_needed_numbers_for_object(object_name):

    num = number_of_samples_per_object - get_sampled_numbers_for_object(object_name)
    return (num)

def get_needed_overall_numbers():
    sum=0
    for o in objects:
        sum+=get_needed_numbers_for_object(o)
    return(sum)

def print_progress():
    for o in objects:
        print " For " + o + " - samples needed: " + str(get_needed_numbers_for_object(o))
        " - samples finished: " + str(get_sampled_numbers_for_object(o))


#Print out was is still needed
print "\n\nWe still need the following samples:"

print_progress()

if get_needed_overall_numbers()>0:
    print "\n\nOverall there are still " +str(get_needed_overall_numbers()) + " samples needed.\n\n"
else:
    print "\n\nAll samples recorded. Thank you. If you needed more, adapt the number_of_samples_per_object parameter on the programm.\n\n"
    exit(0)


# Instructions for the experimenter. Brig the robot in Initial position
print "\n Please put the robot in position. Right arm on the table. Left arm hanging down. Give RETURN after finished.\n"
raw_input()

#Optoforce_sensor
optoforce_sensor = optoforce(ser_number=None, cache_frequency=30)

#Put the left arm in defined position
robot = Motion.Motion("../../../json/nico_humanoid_legged_minimal_for_multimodal_recordings.json",vrep=False)
mover_path = "../../../moves_and_positions/"
mov = Mover.Mover(robot, stiff_off=False)



sleep(4)

#set the robot to be compliant
robot.disableTorqueAll()

robot.openHand('RHand', fractionMaxSpeed=fMS_hand)

robot.enableForceControl("r_wrist_z", 50)
robot.enableForceControl("r_wrist_x", 50)

robot.enableForceControl("r_indexfingers_x", 50)
robot.enableForceControl("r_thumb_x", 50)

# enable torque of left arm joints
robot.enableForceControl("head_z", 20)
robot.enableForceControl("head_y", 20)

robot.enableForceControl("r_shoulder_z", 20)
robot.enableForceControl("r_shoulder_y", 20)
robot.enableForceControl("r_arm_x", 20)
robot.enableForceControl("r_elbow_y", 20)

robot.setAngle("r_thumb_x",160,0.4)
robot.setAngle("r_indexfingers_x",-160,0.4)


# Instructions for the experimenter. Brig the robot in Initial position
print "\n OK. The robot is positioned. We will start the experiment now.\n\n"

pulse_device=pulse_audio_recorder.get_pulse_device()
	
res_x=1920
res_y=1080
framerate=30
amount_of_cams=2
logging.getLogger().setLevel(logging.INFO)

#Vision Recording
device = ImageRecorder.get_devices()[0]
ir = leftcam_ImageRecorder(device, res_x, res_y,framerate=framerate,writer_threads=3,pixel_format="UYVY")

if amount_of_cams>=2:
	device2 = ImageRecorder.get_devices()[1]
	ir2 = rightcam_ImageRecorder(device2, res_x, res_y,framerate=framerate,writer_threads=3,pixel_format="UYVY")

sleep(2)
	
#Sound Recording
ar = pulse_audio_recorder.AudioRecorder(audio_channels=2, samplerate=48000, datadir="./.", audio_device=pulse_device)
	
		
try:
	os.mkdir(data_directory+'/'+action)
except OSError:
	pass


while ( get_needed_overall_numbers() > 0 ):
	
	#Select an object
	o=random.choice(objects)
	while (get_needed_numbers_for_object(o)<1):
		o = random.choice(objects)

	print "Randomly chosen object : " + o + "\n"

	print "\n\n Please put the " + o + " onto the table at the marked position. Then press RETURN."
	raw_input()

	dfl = pd.DataFrame(columns=columns)
	dfr = pd.DataFrame(columns=columns)
	
	#!!!!Write Sample data in database
	sql_command = format_str_sample.format(object_name=o, action=action, timecode= datetime.datetime.now().isoformat())
	#print "\n " +sql_command
	cursor.execute(sql_command)
	sample_number=cursor.lastrowid
	
	str_sample_number=str(sample_number)
	
	cur_dir=data_directory+'/'+action+'/'+str_sample_number
	#print "dir: " + cur_dir
	#raw_input()
	
	try:
		os.mkdir(cur_dir)
	except OSError:
		pass

	try:
		os.mkdir(cur_dir+'/camera1')
	except OSError:
		pass

	try:
		os.mkdir(cur_dir+'/camera2')
	except OSError:
		pass

	label=datetime.datetime.today().isoformat()
	#ar.start_recording(label,fname="./"+action+'/'+str_sample_number+label+".wav",dir_name="./audio/")
	ar.start_recording(label,fname=cur_dir+'/'+label+".wav",dir_name="")
	
	ir.start_recording(cur_dir+'/camera1/picture-{}.png')
	if amount_of_cams>=2:
		ir2.start_recording(cur_dir+'/camera2/picture-{}.png')
    
	breaks=[1.5,2,4,2,1.5]
	for n in range(5):
		mov.move_file_position(mover_path + "pos_pull_"+str(n+1)+".csv", subsetfname=mover_path + "subset_right_arm.csv", move_speed=0.05)
		sleep(breaks[n])
	mov.move_file_position(mover_path + "pos_pull_"+str(1)+".csv", subsetfname=mover_path + "subset_right_arm.csv", move_speed=0.05, )
	sleep(10)

	#PULL action:

	# 1) go to start position
	# 2) retract thumb
	# 3) 


	#robot.openHand("RHand", fractionMaxSpeed=0.4)
	#sleep(5)
	#robot.closeHand("RHand", fractionMaxSpeed=0.4)
	#sleep(5)
	#robot.openHand("RHand", fractionMaxSpeed=0.4)

	#Stop and finish camera recordings
	ir.stop_recording()
	if amount_of_cams>=2:
		ir2.stop_recording()
		
	#Stop and write audio recordings
	ar.stop_recording(0)

	answer="dummy"
	while (answer!="R" and answer!=""):
		print ("Has the recording of this sample been succesful or do you want to repeat it ? (R=Repeat) / (Return=Continue) ")
		answer=raw_input()
	
	if answer=="":
		#Write joint data to file
		for df_set in ((cur_dir+"/"+fnl,dfl),(cur_dir+"/"+fnr,dfr)):
			fnp,dfp=df_set
			with open(fnp, 'a') as f:
				dfp.to_csv(f, header=True)

		#commit the database changes
		connection.commit()
	else:
		call(["rm", "-rf", cur_dir])

		#rollback the database changes
		connection.rollback()


connection.close()
print "\n\n Great! I got all samples together! Finishing experiment.\n"
print_progress()

robot=None

#set the robot to be compliant













