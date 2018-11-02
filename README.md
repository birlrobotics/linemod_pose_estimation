# What is Linemod?
Linemod is a technique for pose estimation using object template matching methods. 

## How do I use this package?

`rosrun linemod_pose_estimation linemod_ensenso_detect_3_mult_detect_service_node`

### paramater
```
"/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/data/memoryChip2_ensenso_templates.yml" 
"/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/data/memoryChip2_ensenso_renderer_params.yml" 
"/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/stl/memoryChip2.stl" 92 150 1e-5 0.02 0.05 20 10 4 
"/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/data/cpu_binary_ensenso_templates.yml" 
"/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/data/cpu_binary_ensenso_renderer_params.yml" 
"/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/stl/cpu_binary.stl" 94
```
When running this command, a service for providing object pose is launched. 

## Using robot 
### Connect UR5 
`roslaunch ur_bringup ur5_bringup.launch robot_ip:=192.168.1.102` 
robot_ip should be modified according to ur own robot.
### Connect camera ensenso
`roslaunch ensenso ensenso_bringup.launch`

## Verify 
`rosservice call /linemod_object_pose "object_id: 0"`
