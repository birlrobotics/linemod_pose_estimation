#include <linemod_pose_estimation/rgbdDetector.h>

//ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <linemod_pose_estimation/rgbdDetector.h>
#include <geometry_msgs/Transform.h>

//tf
#include <tf/transform_broadcaster.h>

//#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>

//ork
#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>
//#include <object_recognition_core/common/pose_result.h>
//#include <object_recognition_core/db/ModelReader.h>

//Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/opencv.hpp>


//PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/median_filter.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>

//ensenso
//#include <ensenso/RegistImage.h>
//#include <ensenso/CaptureSinglePointCloud.h>


//boost
#include <boost/foreach.hpp>

//std
#include <math.h>

//time synchronize
#include <message_filters/time_synchronizer.h>
#define APPROXIMATE

#ifdef EXACT
#include <message_filters/sync_policies/exact_time.h>
#endif
#ifdef APPROXIMATE
#include <message_filters/sync_policies/approximate_time.h>
#endif

#ifdef EXACT
typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image,sensor_msgs::PointCloud2> SyncPolicy;
#endif
#ifdef APPROXIMATE
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2> SyncPolicy;
#endif

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

//namespace
using namespace cv;
using namespace std;

bool sortScore(const ClusterData& score1 , const ClusterData& score2)
{
    return(score1.score > score2.score);

}

struct LinemodData{
  LinemodData(
          std::vector<cv::Vec3f> _pts_ref,
          std::vector<cv::Vec3f> _pts_model,
          std::string _match_class,
          int _match_id,
          const float _match_sim,
          const cv::Point location_,
          const float _icp_distance,
          const cv::Matx33f _r,
          const cv::Vec3f _t){
    pts_ref = _pts_ref;
    pts_model = _pts_model;
    match_class = _match_class;
    match_id=_match_id;
    match_sim = _match_sim;
    location=location_;
    icp_distance = _icp_distance;
    r = _r;
    t = _t;
    check_done = false;
  }
  std::vector<cv::Vec3f> pts_ref;
  std::vector<cv::Vec3f> pts_model;
  std::string match_class;
  int match_id;
  float match_sim;
  float icp_distance;
  cv::Matx33f r;
  cv::Vec3f t;
  cv::Point location;
  bool check_done;
};


class linemod_detect
{
    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    ros::Subscriber sub_cam_info;
    //image_transport::Subscriber sub_color_;
    //image_transport::Subscriber sub_depth;
    //image_transport::Publisher pub_color_;
    //image_transport::Publisher pub_depth;
    message_filters::Subscriber<sensor_msgs::Image> sub_color;
    message_filters::Subscriber<sensor_msgs::Image> sub_depth;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_pc;
    message_filters::Synchronizer<SyncPolicy> sync;
    //message_filters::TimeSynchronizer<sensor_msgs::Image,sensor_msgs::Image,sensor_msgs::PointCloud2> timeSync;
    ros::Publisher pc_rgb_pub_;
    ros::Publisher extract_pc_pub;
    ros::Publisher object_pose;

    //voting space
    unsigned int* accumulator;
    uchar clustering_step_;

public:
    Ptr<linemod::Detector> detector;
    float threshold;
    bool is_K_read;
    cv::Vec3f T_ref;

    //Params
    vector<Mat> Rs_,Ts_;
    vector<double> Distances_;
    vector<double> Obj_origin_dists;
    vector<Mat> Ks_;
    vector<Rect> Rects_;
    Mat K_depth;
    Matx33d K_rgb;
    Matx33f R_diag;
    int renderer_n_points;
    int renderer_angle_step;
    double renderer_radius_min;
    double renderer_radius_max;
    double renderer_radius_step;
    int renderer_width;
    int renderer_height;
    double renderer_focal_length_x;
    double renderer_focal_length_y;
    double renderer_near;
    double renderer_far;
    Renderer3d *renderer_;
    RendererIterator *renderer_iterator_;
    float linemod_match_threshold;
    float nms_radius;
    float collision_rate_threshold;
    rgbdDetector::IMAGE_WIDTH image_width;
    int bias_x;
    bool is_nms_neighbor_set;
    bool is_hypothesis_verification_set;

    float px_match_min_;
    float icp_dist_min_;
    float orientation_clustering_th_;

    std::string depth_frame_id_;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    //std::vector <object_recognition_core::db::ObjData> objs_;
    std::vector<LinemodData> objs_;//pose result after clustering process
    std::vector<LinemodData> final_poses;//pose result after clustering process

    tf::TransformBroadcaster tf_broadcaster;
    //Service client
//    ros::ServiceClient ensenso_registImg_client;
//    ros::ServiceClient ensenso_singlePc_client;

    //File path for ground truth
    std::string gr_prefix;

    //Wrapped detector
    rgbdDetector rgbd_detector;

    vector<ClusterData> cluster_data;

    Eigen::Matrix3d orient_modifying_matrix;

    int count_test ;

    int waitTime ;


public:
        linemod_detect(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold, float icp_maxCorresDist, uchar clustering_step,float orientation_clustering_th):
            it(nh),
            gr_prefix("/home/yake/catkin_ws/src/linemod_pose_est/dataset/Annotation/"),
            sub_color(nh,"/camera/rgb/image_rect_color",5),
            sub_depth(nh,"/camera/depth_registered/image_raw",5),
            sub_pc(nh,"/camera/depth_registered/points",5),
            depth_frame_id_("camera_link"),
            sync(SyncPolicy(10), sub_color, sub_depth,sub_pc),
            px_match_min_(0.25f),
            icp_dist_min_(0.06f),
            clustering_step_(clustering_step),
            linemod_match_threshold(89),
            nms_radius(4),
            collision_rate_threshold(0.3),
            image_width(rgbdDetector::CARMINE),
            bias_x(0),
            is_hypothesis_verification_set(false),
            is_nms_neighbor_set(false),
            count_test(0),
            waitTime(1)

        {
            //Publisher
            //pub_color_=it.advertise ("/sync_rgb",2);
            //pub_depth=it.advertise ("/sync_depth",2);
            pc_rgb_pub_=nh.advertise<sensor_msgs::PointCloud2>("ensenso_rgb_pc",1);
            extract_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/render_pc",1);
            object_pose = nh.advertise<geometry_msgs::Transform>("object_pose",1);

            //the intrinsic matrix
            sub_cam_info=nh.subscribe("/camera/depth/camera_info",1,&linemod_detect::read_cam_info,this);

            //linemod detect thresh
            linemod_match_threshold=detect_score_threshold;

            //read the saved linemod detecor
            detector=readLinemod (template_file_name);

            //read the poses of templates
            readLinemodTemplateParams (renderer_params_name,Rs_,Ts_,Distances_,Obj_origin_dists,
                                       Ks_,Rects_,renderer_n_points,renderer_angle_step,
                                       renderer_radius_min,renderer_radius_max,
                                       renderer_radius_step,renderer_width,renderer_height,
                                       renderer_focal_length_x,renderer_focal_length_y,
                                       renderer_near,renderer_far);


            //load the stl model to GL renderer
            renderer_ = new Renderer3d(mesh_path);
            renderer_->set_parameters(renderer_width, renderer_height, renderer_focal_length_x, renderer_focal_length_y, renderer_near, renderer_far);
            renderer_iterator_ = new RendererIterator(renderer_, renderer_n_points);
            renderer_iterator_->angle_step_ = renderer_angle_step;
            renderer_iterator_->radius_min_ = float(renderer_radius_min);
            renderer_iterator_->radius_max_ = float(renderer_radius_max);
            renderer_iterator_->radius_step_ = float(renderer_radius_step);

            icp.setMaximumIterations (icp_max_iter);
            icp.setMaxCorrespondenceDistance (icp_maxCorresDist);
            icp.setTransformationEpsilon (icp_tr_epsilon);
            icp.setEuclideanFitnessEpsilon (icp_fitness_threshold);


            R_diag=Matx<float,3,3>(1.0,0.0,0.0,
                                  0.0,1.0,0.0,
                                  0.0,0.0,1.0);
            K_rgb=Matx<double,3,3>(535.566011,0.0,320,
                                   0.0,535.566011,240,
                                   0.0,0.0,1.0);

            //Service client
//            ensenso_registImg_client=nh.serviceClient<ensenso::RegistImage>("grab_registered_image");
//           ensenso_singlePc_client=nh.serviceClient<ensenso::CaptureSinglePointCloud>("capture_single_point_cloud");

            //Orientation Clustering threshold
            orientation_clustering_th_=orientation_clustering_th;

            //Subscribe to rgb image topic and depth image topic
            sync.registerCallback(boost::bind(&linemod_detect::detect_cb,this,_1,_2,_3));

        }

        virtual ~linemod_detect()
        {
            if(accumulator)
                free(accumulator);
        }

        void detect_cb(const sensor_msgs::ImageConstPtr& msg_rgb , const sensor_msgs::ImageConstPtr& msg_depth,const sensor_msgs::PointCloud2ConstPtr& msg_pc2)
        {
            cluster_data.clear();

            //Read camera intrinsic params
            if(!is_K_read)
                return;

            //Check if non-maximum suppression or hypothesis verification set
            if(!is_nms_neighbor_set || !is_hypothesis_verification_set)
                return;

            //If LINEMOD detector is not loaded correctly, return
            if(detector->classIds ().empty ())
            {
                ROS_INFO("Linemod detector is empty");
                return;
            }

            PointCloudXYZ::Ptr pc_ptr(new PointCloudXYZ);
            pcl::fromROSMsg(*msg_pc2,*pc_ptr);

            //Get LINEMOD source image
            vector<Mat> sources;
            rosMsgs_to_linemodSources(msg_rgb,msg_depth,sources);
            Mat mat_rgb,mat_depth;
            mat_rgb=sources[0];
            mat_depth=sources[1];

            cv::imwrite("/home/yake/test.jpg",mat_rgb);

            //Image for displaying detection
            Mat initial_img=mat_rgb.clone();
            Mat display=mat_rgb.clone();
            Mat final=mat_rgb.clone();
            Mat cluster_img=mat_rgb.clone();
            Mat cluster_filter_img=mat_rgb.clone();
            Mat nms_img = mat_rgb.clone();

            //Perform the LINEMOD detection
            std::vector<linemod::Match> matches;
            double t=cv::getTickCount ();
            rgbd_detector.linemod_detection(detector,sources,linemod_match_threshold,matches);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by template matching: "<<t<<" s"<<endl;

            //Display all the results
                //Contour
            for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++)
            {
                std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
                drawResponse(templates, 1, display,cv::Point(it->x,it->y), 2);
            }
//            imshow("intial results",display);
//            waitKey(waitTime);
                //Rectangle
//            for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++)
//            {
//                cv::Rect rect_tmp= Rects_[it->template_id];
//                rect_tmp.x=it->x;
//                rect_tmp.y=it->y;
//                rectangle(initial_img,rect_tmp,Scalar(0,0,255),2);
//            }

            //Clustering based on Row Col Depth
            std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
            t=cv::getTickCount ();
            //rcd_voting(vote_row_col_step, vote_depth_step, matches,map_match, voting_height_cells, voting_width_cells);
            rgbd_detector.rcd_voting(Obj_origin_dists,renderer_radius_min,clustering_step_,renderer_radius_step,matches,map_match);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            //cout<<"Time consumed by rcd voting: "<<t<<" s"<<endl;
            //Display all the results
            RNG rng(0xFFFFFFFF);
            for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();it != map_match.end();it++)
            {
                Scalar color(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
                for(std::vector<linemod::Match>::iterator it_v= it->second.begin();it_v != it->second.end();it_v++)
                {
                    cv::Rect rect_tmp= Rects_[it_v->template_id];
                    rect_tmp.x=it_v->x;
                    rect_tmp.y=it_v->y;
                    rectangle(cluster_img,rect_tmp,color,2);
                }
            }
//            imshow("cluster",cluster_img);
//            cv::waitKey (0);


            //Filter based on size of clusters
            uchar thresh=0;
            t=cv::getTickCount ();
            rgbd_detector.cluster_filter(map_match,thresh);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by cluster filter: "<<t<<" s"<<endl;
            for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();it != map_match.end();it++)
            {
                Scalar color(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
                for(std::vector<linemod::Match>::iterator it_v= it->second.begin();it_v != it->second.end();it_v++)
                {
                    cv::Rect rect_tmp= Rects_[it_v->template_id];
                    rect_tmp.x=it_v->x;
                    rect_tmp.y=it_v->y;
                    rectangle(cluster_filter_img,rect_tmp,color,2);
                }
            }
//            imshow("cluster filtered",cluster_filter_img);
//            cv::waitKey (waitTime);

            //Use match similarity score as evaluation
            //rgbd_detector.cluster_scoring(map_match,mat_depth,cluster_data);
            Matx33d K_tmp=Matx<double,3,3>(535.566011, 0.0, 319.5000,
                                           0.0, 537.168115, 239.5000,
                                           0.0, 0.0, 1.0);
            Mat depth_tmp=mat_depth.clone();
            t=cv::getTickCount ();
            rgbd_detector.cluster_scoring(renderer_iterator_,K_tmp,Rs_,Ts_,map_match,depth_tmp,cluster_data);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by scoring: "<<t<<endl;


            //Non-maxima suppression
            t=cv::getTickCount ();
            rgbd_detector.nonMaximaSuppressionUsingIOU(cluster_data,nms_radius,Rects_,map_match);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by non-maxima suppression: "<<t<<endl;
            //Display
            for(vector<ClusterData>::iterator iter=cluster_data.begin();iter!=cluster_data.end();++iter)
            {
              for(std::vector<linemod::Match>::iterator it= iter->matches.begin();it != iter->matches.end();it++)
              {
                  std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
                  drawResponse(templates, 1, nms_img,cv::Point(it->x,it->y), 2);
              }
            }
//            imshow("Non-maxima suppression",nms_img);
//            cv::waitKey (waitTime);


            //Pose average
            t=cv::getTickCount ();
            rgbd_detector.getRoughPoseByClustering(cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by rough pose estimation : "<<t<<endl;
//            vizResultPclViewer(cluster_data,pc_ptr);


            //Pose refinement
            t=cv::getTickCount ();
            rgbd_detector.icpPoseRefine(cluster_data,icp,pc_ptr,image_width,bias_x,false);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose refinement: "<<t<<endl;


            //Hypothesis verification
            t=cv::getTickCount ();
            rgbd_detector.hypothesisVerification(cluster_data,0.004,collision_rate_threshold,false);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by hypothesis verification: "<<t<<endl;
            //Display all the bounding box
//            for(int ii=0;ii<cluster_data.size();++ii)
//            {
            if(cluster_data.size()!=0)
            {
                std::sort(cluster_data.begin(),cluster_data.end(),sortScore);
                rectangle(final,cluster_data[0].rect,Scalar(0,0,255),2);
            }

//            }

//            for(int i=0;i<cluster_data.size();++i)
//            {
//              templateRefinement(cluster_data[i],pc_ptr);
//            }

//            templateRefinement(cluster_data[0],pc_ptr);

            //pose publish
            if(cluster_data.size()!=0)
            {
//                for(int i =0; i <  cluster_data.size(); ++i)
//                  {
                        Eigen::Vector4d R_quat_vec(0.0,0.0,0.0,0.0);
                        Eigen::Vector3d T_vec (0.0,0.0,0.0);
                        Eigen::Matrix3d R_temp =cluster_data[0].pose.linear();
                        if(R_temp(2,2) < 0)  //z aixs is always down
                          {
                             R_temp.col(0) = -R_temp.col(0);
                             R_temp.col(2) = -R_temp.col(2);
                          }
                        cluster_data[0].pose.linear() = R_temp ;
                        R_quat_vec=Eigen::Quaterniond(R_temp).coeffs();
                        T_vec = cluster_data[0].pose.translation();
                        std::cerr<<" T_vec = "<<endl<<T_vec<<endl;
                        std::cerr<<" R_quat_vec = "<<endl<<R_quat_vec<<endl; //maybe qw qx qy qz
                        std::cerr<<" R 3x3 = "<<endl<<R_temp<<endl;


                        geometry_msgs::Transform pose_tf;
                        pose_tf.translation.x = T_vec[0];
                        pose_tf.translation.y = T_vec[1];
                        pose_tf.translation.z = T_vec[2];
                        pose_tf.rotation.w = R_quat_vec[0];
                        pose_tf.rotation.x = R_quat_vec[1];
                        pose_tf.rotation.y = R_quat_vec[2];
                        pose_tf.rotation.z = R_quat_vec[3];
                        object_pose.publish(pose_tf);
//                   }
            }


            //Viz
//            vizResultPclViewer(cluster_data,pc_ptr);
            imshow("final result",final);
            cv::waitKey (waitTime);

//            ros::shutdown();

        }

        void vizResultPclViewer(const vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc_ptr)
        {
            pcl::visualization::PCLVisualizer view("vvvvv");
            //view.addPointCloud(pc_ptr,"scene");
            for(int ii=0;ii<cluster_data.size();++ii)
            {
                pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(cluster_data[ii].model_pc);
                string str="model ";
                string str2="scene ";
                stringstream ss;
                ss<<ii;
                str+=ss.str();
                str2+=ss.str();
                view.addPointCloud(cluster_data[ii].model_pc,color,str);
                //view.addPointCloud(cluster_data[ii].scene_pc,str2);

                //Get eigen tf
//                Eigen::Affine3d obj_pose;
//                Eigen::Matrix3d rot;
//                cv2eigen(cluster_data[ii].orientation,rot);
//                obj_pose.linear()=rot;
//                obj_pose.translation()<<cluster_data[ii].position[0],cluster_data[ii].position[1],cluster_data[ii].position[2];

                Eigen::Affine3f obj_pose_f=cluster_data[ii].pose.cast<float>();
                view.addCoordinateSystem(0.08,obj_pose_f);
            }
            view.addPointCloud(pc_ptr,"scene points");
            view.spin();
        }

        void templateRefinement(ClusterData object,PointCloudXYZ::Ptr pc_ptr)
        {
            //Get World--->Camera transform(camera pose), eye point up vector...
            Eigen::Affine3d object_pose = object.pose;
            Eigen::Affine3d c2_camera_pose = object_pose.inverse();

            //Eye position
            cv::Vec3d eye(c2_camera_pose.translation()[0], c2_camera_pose.translation()[1], c2_camera_pose.translation()[2]);
            Eigen::Matrix3d camera_orientation = c2_camera_pose.linear();
            //Up vector
            cv::Vec3d up(-camera_orientation(0,1),-camera_orientation(1,1),-camera_orientation(2,1));
            //Look at point
            cv::Vec3d look_at_point = get_look_at_point(pc_ptr,c2_camera_pose);
            //cv::Vec3d look_at_point(0.0,0.0,0.0);

            //Render
            cv::Mat depth_image, mask,flip_depth_image,flip_mask;
            cv::Rect rect;
            renderer_iterator_->renderDepthOnly(depth_image, mask, rect, eye, up,look_at_point);
            cv::flip(depth_image,flip_depth_image,0);
            cv::flip(mask,flip_mask,0);
            cv::imshow("template refine",flip_mask);
            cv::imwrite("/home/yake/template_refine.jpg",flip_mask);
            waitKey(0);

            Mat K_matrix=(Mat_<double>(3,3)<<renderer_focal_length_x,0.0,depth_image.cols/2,
                                          0.0,renderer_focal_length_y,depth_image.rows/2,
                                          0.0,0.0,1.0);
            Mat pc_cv;
            cv::depthTo3d(flip_depth_image,K_matrix,pc_cv);    //mm ---> m
            PointCloudXYZ::Ptr refined_model_pc(new PointCloudXYZ);
            refined_model_pc->header.frame_id="/camera_link";
            for(int ii=0;ii<pc_cv.rows;++ii)
            {
                double* row_ptr=pc_cv.ptr<double>(ii);
                for(int jj=0;jj<pc_cv.cols;++jj)
                {
                    double* data =row_ptr+jj*3;
                    if(std::isnan(data[0]) || std::isnan(data[1]) || std::isnan(data[2]))
                      continue;
                    refined_model_pc->points.push_back(pcl::PointXYZ(data[0],data[1],data[2]));
                }
            }
            std::vector<int> index1;
            PointCloudXYZ::Ptr pc_ptr_(new PointCloudXYZ);
            pcl::removeNaNFromPointCloud(*refined_model_pc,*refined_model_pc,index1);
            pcl::removeNaNFromPointCloud(*pc_ptr,*pc_ptr_,index1);
            refined_model_pc->height=1;
            refined_model_pc->width=refined_model_pc->points.size();

            //ICP refine using new point cloud
            //Coarse alignment
            icp.setMaxCorrespondenceDistance(0.05);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setMaximumIterations(50);
            icp.setRANSACOutlierRejectionThreshold(0.01);
            icp.setInputSource (refined_model_pc);
            icp.setInputTarget (pc_ptr_);
            icp.align (*(refined_model_pc));
            if(!icp.hasConverged())
            {
                cout<<"ICP cannot converge"<<endl;
            }
            else{
                //cout<<"ICP fitness score of coarse alignment: "<<icp.getFitnessScore()<<endl;
            }

            //Save pose and model points before icp
            Eigen::Affine3d origin_object_pose = object.pose;
            PointCloudXYZ::Ptr origin_model_pc(new PointCloudXYZ);
            origin_model_pc=object.model_pc;

            //Update poseamera_pose
            Eigen::Matrix4f tf_mat = icp.getFinalTransformation();
            Eigen::Matrix4d tf_mat_d=tf_mat.cast<double>();
            Eigen::Affine3d tf(tf_mat_d);
            object.pose=tf*object.pose;

            //Viz
            pcl::visualization::PCLVisualizer v("after template refinment");
            pcl::visualization::PCLVisualizer v2("before template refinment");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> refined_model_pc_color(refined_model_pc,0,255,0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_pc_color(origin_model_pc,255,0,0);

//            v.addPointCloud(refined_model_pc,refined_model_pc_color,"refined model");
//            //v.addPointCloud(origin_model_pc,model_pc_color,"model");
//            v.addPointCloud(pc_ptr);
//            v.addCoordinateSystem(0.2,object.pose.cast<float>());
//            v.spinOnce();

//            v2.addPointCloud(pc_ptr);
//            v2.addCoordinateSystem(0.2,origin_object_pose.cast<float>());
//            v2.addPointCloud(origin_model_pc,model_pc_color,"model");
//            v2.spin();

            object.model_pc = refined_model_pc;

        }

        cv::Vec3d get_look_at_point(PointCloudXYZ::Ptr scene_pc,Eigen::Affine3d transform)
        {
            int col = scene_pc->width/2;
            int row = scene_pc->height/2;
            pcl::PointXYZ look_at_point= scene_pc->at(col,row);
            while(isnan(look_at_point.x) || isnan(look_at_point.y) || isnan(look_at_point.z))
            {
                col++;
                look_at_point = scene_pc->at(col,row);
            }
            Eigen::Vector3d look_at_point_;
            look_at_point_[0] = look_at_point.x;
            look_at_point_[1] = look_at_point.y;
            look_at_point_[2] = look_at_point.z;
            look_at_point_ = transform * look_at_point_;
            cv::Vec3d look_at_point_cv(look_at_point_[0],look_at_point_[1],look_at_point_[2]);
            return look_at_point_cv;
        }

        static cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename)
        {

          //cv::Ptr<cv::linemod::Detector> detector = cv::makePtr<cv::linemod::Detector>();
          cv::Ptr<cv::linemod::Detector> detector(new cv::linemod::Detector);
          cv::FileStorage fs(filename, cv::FileStorage::READ);
          detector->read(fs.root());

          cv::FileNode fn = fs["classes"];
          for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
            detector->readClass(*i);

          return detector;
        }

        void drawResponse(const std::vector<cv::linemod::Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset,
                     int T)
        {
          static const cv::Scalar COLORS[5] =
          { CV_RGB(0, 0, 255), CV_RGB(0, 255, 0), CV_RGB(255, 255, 0), CV_RGB(255, 140, 0), CV_RGB(255, 0, 0) };
          if (dst.channels() == 1)
            cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);

          //cv::circle(dst, cv::Point(offset.x + 20, offset.y + 20), T / 2, COLORS[4]);
          if (num_modalities > 5)
            num_modalities = 5;
          for (int m = 0; m < num_modalities; ++m)
          {
        // NOTE: Original demo recalculated max response for each feature in the TxT
        // box around it and chose the display color based on that response. Here
        // the display color just depends on the modality.
            cv::Scalar color = COLORS[m+2];

            for (int i = 0; i < (int) templates[m].features.size(); ++i)
            {
              cv::linemod::Feature f = templates[m].features[i];
              cv::Point pt(f.x + offset.x, f.y + offset.y);
              cv::circle(dst, pt, T / 2, color);
            }
          }
        }

         void readLinemodTemplateParams(const std::string fileName,
                                        std::vector<cv::Mat>& Rs,
                                        std::vector<cv::Mat>& Ts,
                                        std::vector<double>& Distances,
                                        std::vector<double>& Obj_origin_dists,
                                        std::vector<cv::Mat>& Ks,
                                        std::vector<cv::Rect>& Rects,
                                        int& renderer_n_points,
                                        int& renderer_angle_step,
                                        double& renderer_radius_min,
                                        double& renderer_radius_max,
                                        double& renderer_radius_step,
                                        int& renderer_width,
                                        int& renderer_height,
                                        double& renderer_focal_length_x,
                                        double& renderer_focal_length_y,
                                        double& renderer_near,
                                        double& renderer_far)
        {
            FileStorage fs(fileName,FileStorage::READ);

            for(int i=0;;++i)
            {
                std::stringstream ss;
                std::string s;
                s="Template ";
                ss<<i;
                s+=ss.str ();
                FileNode templates = fs[s];
                if(!templates.empty ())
                {
                    Mat R_tmp,T_tmp,K_tmp;
                    Rect rect_tmp;
                    float D_tmp,obj_dist_tmp;
                    templates["R"]>>R_tmp;
                    Rs.push_back (R_tmp);
                    templates["T"]>>T_tmp;
                    Ts.push_back (T_tmp);
                    templates["K"]>>K_tmp;
                    Ks.push_back (K_tmp);
                    templates["D"]>>D_tmp;
                    Distances.push_back (D_tmp);
                    templates["Ori_dist"] >>obj_dist_tmp;
                    Obj_origin_dists.push_back (obj_dist_tmp);
                    templates["Rect"]>>rect_tmp;
                    Rects.push_back(rect_tmp);

                }
                else
                {
                    //fs["K Intrinsic Matrix"]>>K_matrix;
                    //std::cout<<K_matrix<<std::endl;INFO]
                    fs["renderer_n_points"]>>renderer_n_points;
                    fs["renderer_angle_step"]>>renderer_angle_step;
                    fs["renderer_radius_min"]>>renderer_radius_min;
                    fs["renderer_radius_max"]>>renderer_radius_max;
                    fs["renderer_radius_step"]>>renderer_radius_step;
                    fs["renderer_width"]>>renderer_width;
                    fs["renderer_height"]>>renderer_height;
                    fs["renderer_focal_length_x"]>>renderer_focal_length_x;
                    fs["renderer_focal_length_y"]>>renderer_focal_length_y;
                    fs["renderer_near"]>>renderer_near;
                    fs["renderer_far"]>>renderer_far;
                    break;
                }
            }

            fs.release ();
        }

         void read_cam_info(const sensor_msgs::CameraInfoConstPtr& infoMsg)
         {
             K_depth = (cv::Mat_<float>(3, 3) <<
                                  infoMsg->K[0], infoMsg->K[1], infoMsg->K[2],
                                  infoMsg->K[3], infoMsg->K[4], infoMsg->K[5],
                                  infoMsg->K[6], infoMsg->K[7], infoMsg->K[8]);

             Mat distCoeffs = (cv::Mat_<double>(1, 5) <<
                                   infoMsg->D[0],
                                   infoMsg->D[1],
                                   infoMsg->D[2],
                                   infoMsg->D[3],
                                   infoMsg->D[4]);
             sub_cam_info.shutdown ();
             is_K_read=true;
         }         

       void rosMsgs_to_linemodSources(const sensor_msgs::ImageConstPtr& msg_rgb , const sensor_msgs::ImageConstPtr& msg_depth,vector<Mat>& sources)
        {
            //Convert ros msg to OpenCV mat
            cv_bridge::CvImagePtr img_ptr_rgb;
            cv_bridge::CvImagePtr img_ptr_depth;
            Mat mat_rgb;
            Mat mat_depth;

            try{
                 img_ptr_depth = cv_bridge::toCvCopy(*msg_depth);
             }
             catch (cv_bridge::Exception& e)
             {
                 ROS_ERROR("cv_bridge exception:  %s", e.what());
                 return;
             }

             try{
                 img_ptr_rgb = cv_bridge::toCvCopy(*msg_rgb, sensor_msgs::image_encodings::BGR8);
                 mat_rgb=img_ptr_rgb->image.clone();
             }
             catch (cv_bridge::Exception& e)
             {
                 ROS_ERROR("cv_bridge exception:  %s", e.what());
                 return;
             }

             if(img_ptr_depth->image.depth ()==CV_32F)
             {
                img_ptr_depth->image.convertTo (mat_depth,CV_16UC1,1000.0);
             }
             else
             {
                 img_ptr_depth->image.copyTo(mat_depth);
             }

             sources.push_back(mat_rgb);
             sources.push_back(mat_depth);
        }

        void setTemplateMatchingThreshold(float threshold_)
        {
            linemod_match_threshold = threshold_;
        }

        void setNonMaximumSuppressionRadisu(double radius)\
        {
            nms_radius=radius;
            is_nms_neighbor_set=true;
        }

        void setHypothesisVerficationThreshold(float threshold_)
        {
            collision_rate_threshold = threshold_;
            is_hypothesis_verification_set=true;
        }

        void setOrientationModifyingMatrix(Eigen::Matrix3d modifying_mat)
        {
            orient_modifying_matrix=modifying_mat;
        }

};

int main(int argc,char** argv)
{
    ros::init (argc,argv,"linemod_detect");
    std::string linemod_template_path;
    std::string renderer_param_path;
    std::string model_stl_path;
    float detect_score_th;
    int icp_max_iter;
    float icp_tr_epsilon;
    float icp_fitness_th;
    float icp_maxCorresDist;
    uchar clustering_step;
    float orientation_clustering_step;
    float nms_neighbor;
    float hv_thresh;
    Eigen::Vector3d rot_axis(0.0,0.0,1.0);
    float orientation_modifying_degree;

    linemod_template_path=argv[1];
    renderer_param_path=argv[2];
    model_stl_path=argv[3];
    detect_score_th=atof(argv[4]);
    icp_max_iter=atoi(argv[5]);
    icp_tr_epsilon=atof(argv[6]);
    icp_fitness_th=atof(argv[7]);
    icp_maxCorresDist=atof(argv[8]);
    clustering_step=atoi(argv[9]);
    orientation_clustering_step=atof(argv[10]);
    nms_neighbor=atof(argv[11]);
    hv_thresh=0.5;
    orientation_modifying_degree=3.14;


//    string cmd;
//    while(1)
//      {
//         cout<<"strat a new detection? input [y] to begin, or [n] to quit. "<<endl;
//         cin>>cmd;
//         if(cmd=="y")
//           {
           linemod_detect detector(linemod_template_path,renderer_param_path,model_stl_path,
                                   detect_score_th,icp_max_iter,icp_tr_epsilon,icp_fitness_th,icp_maxCorresDist,clustering_step,orientation_clustering_step);
           detector.setOrientationModifyingMatrix(Eigen::AngleAxisd(orientation_modifying_degree,rot_axis).matrix());
           detector.setHypothesisVerficationThreshold(hv_thresh);
           detector.setNonMaximumSuppressionRadisu(nms_neighbor);
           ros::spin();

//           }
//         else if (cmd == "n")
//           break;
//      }

    std::cout<<"test ------------0------------"<<endl;
    return 0;
}
