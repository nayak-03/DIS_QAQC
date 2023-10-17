import os
import json
import PySpin 
import itertools


system = PySpin.System.GetInstance()

cam_list = system.GetCameras()
num_cameras = cam_list.GetSize()

print('Number of cameras detected: %d' % num_cameras)

# Finish if there are no cameras or more than one camera
if num_cameras != 1:
	# Clear camera list before releasing system
	cam_list.Clear()

	# Release system instance
	system.ReleaseInstance()
	print('This program works only on one camera at a time :)')
	exit()

# grab the settings we want :)
input_file = input("Please enter the name of the configuration file you'd like to use :)\n")

with open (input_file, "r") as config_file:
	configs = json.load(config_file)
	
if "Num Images" in configs: 
	print("\n")
	print("Taking %d images for each test" % configs["Num Images"])
	print(len(configs) - 1, " tests in this configuration file \n")
else: 
	print("incorrectly formatted config file, please try again")
	cam_list.Clear()
	system.ReleaseInstance()
	exit()

# grab only the tests from the config file 
tests = dict(itertools.islice(configs.items(), 1, None))
print(tests)

cam = cam_list.GetByIndex(0)
# Initialize camera
cam.Init()
# Retrieve nodemap
nodemap = cam.GetNodeMap()




# do the camera setup: acquitsition mode cont., exposure not auto, gain not auto :) 
# 
# acquisition mode cont.
node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
	print('Unable to set acquisition mode (enum retrieval). Aborting...')
	cam.DeInit()
	cam = None
	cam_list.Clear()
	system.ReleaseInstance()
	
node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
if not PySpin.IsReadable(node_acquisition_mode_continuous):
	print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
	cam.DeInit()
	cam = None
	cam_list.Clear()
	system.ReleaseInstance()
	
# Retrieve integer value from entry node
acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

# Set integer value from entry node as new value of enumeration node
node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

# turn off exposure auto mode 
node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
if not PySpin.IsReadable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
	print('Unable to turn off auto exposure mode (enum retrieval). Aborting...')
	cam.DeInit()
	cam = None
	cam_list.Clear()
	system.ReleaseInstance()
	
node_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
if not PySpin.IsReadable(node_exposure_auto_off):
	print('Unable to set turn off auto exposure mode (entry retrieval). Aborting...')
	cam.DeInit()
	cam = None
	cam_list.Clear()
	system.ReleaseInstance()
	
# Retrieve integer value from entry node
exposure_auto_off = node_exposure_auto_off.GetValue()

# Set integer value from entry node as new value of enumeration node
node_exposure_auto.SetIntValue(exposure_auto_off)

### ACQUIRE IMAGES ### 

# iterate through all tests and take num images amount of images :) 
for test in tests: 
	# set the test parameters (can add more if needed) 
	# TODO: figure out why this does not work >:( 
	""" 
	node_exposure = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureTime'))
	if not PySpin.IsReadable(node_exposure): #or not PySpin.IsWritable(node_exposure):
		print('Unable to set exposure (enum retrieval). Aborting...')
		cam.DeInit()
		cam = None
		cam_list.Clear()
		system.ReleaseInstance()
		exit()
	
	exit()   
	node_exposure.SetValue(test["Exposure"])
	"""
	
	
	"""
	node_gain = PySpin.CEnumerationPtr(nodemap.GetNode('Gain'))
	if not PySpin.IsReadable(node_gain) or not PySpin.IsWritable(node_gain):
		print('Unable to set gain (enum retrieval). Aborting...')
		cam.DeInit()
		cam = None
		cam_list.Clear()
		system.ReleaseInstance()
		exit()
	    
	node_gain.SetValue(test["Gain"])
	""" 
	
	cam.BeginAcquisition()
	
	for i in range(configs["Num Images"]):
		image_result = cam.GetNextImage()
		
		if image_result.IsIncomplete():
			print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

		else:
			# print image information (required or not?)
		    	width = image_result.GetWidth()
		    	height = image_result.GetHeight()
		    	print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))
			

		#TODO create a descriptive file name
		filename = 'dark_image_test_5/dark_image_test_5_%04d.jpg' % i
		image_result.Save(filename)
		print('Image saved at %s' % filename)

		#  Release image
		image_result.Release()
		print('')
	
	cam.EndAcquisition()

	
cam.DeInit()
cam = None
cam_list.Clear()
system.ReleaseInstance()




