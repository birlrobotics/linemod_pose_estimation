××××××××××××××××××   linemod_ensenso_detect_3_mult_detect_service.cpp   ××××××××××××××××××××××××××××

1`rosrun linemod_pose_estimation linemod_ensenso_detect_3_mult_detect_service_node
"/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/data/memoryChip2_ensenso_templates.yml" "/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/data/memoryChip2_ensenso_renderer_params.yml" "/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/stl/memoryChip2.stl" 92 150 1e-5 0.02 0.05 20 10 4 "/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/data/cpu_binary_ensenso_templates.yml" "/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/data/cpu_binary_ensenso_renderer_params.yml" "/home/birl-spai-ubuntu14/birl_ws/src/csl_vision/linemod_pose_estimation/config/stl/cpu_binary.stl" 94

2`使用时，先把电脑连上UR5，在运行 roslaunch ur_bringup ur5_bringup.launch robot_ip:=192.168.1.102和roslaunch ensenso ensenso_bringup.launch；订阅service "/linemod_object_pose"即可，消息类型 geometry_msgs::Transform。

使用rosservice call /linemod_object_pose "object_id: 0"来验证service是否成功；“0”表示内存条，“1”表示CPU；







