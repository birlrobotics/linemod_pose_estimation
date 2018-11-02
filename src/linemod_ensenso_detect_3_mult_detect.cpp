#include <linemod_pose_estimation/rgbdDetector.h>
#include <pcl/visualization/pcl_visualizer.h>
//ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>

//ork
#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>

//ensenso
#include <ensenso/RegistImage.h>
#include <ensenso/CaptureSinglePointCloud.h>

//boost
#include <boost/foreach.hpp>

//std
#include <math.h>

//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//time synchronize
#define APPROXIMATE

#ifdef EXACT
#include <message_filters/sync_policies/exact_time.h>
#endif
#ifdef APPROXIMATE
#include <message_filters/sync_policies/approximate_time.h>
#endif

#ifdef EXACT
typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
#endif
#ifdef APPROXIMATE
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
#endif

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

//namespace
using namespace cv;
using namespace std;

bool sortPointSize(const vector<Point>& contours1,const vector<Point>& contours2)
{
    return(contours1.size()>contours2.size());
}

struct LinemodData
{
  LinemodData
         (
          std::vector<cv::Vec3f> _pts_ref,
          std::vector<cv::Vec3f> _pts_model,
          std::string _match_class,
          int _match_id,
          const float _match_sim,
          const cv::Point location_,
          const float _icp_distance,
          const cv::Matx33f _r,
          const cv::Vec3f _t
          )
  {
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
    message_filters::Synchronizer<SyncPolicy> sync;
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
    ros::ServiceClient ensenso_registImg_client;
    ros::ServiceClient ensenso_singlePc_client;

    //Offset for cropped image
    int bias_x;
    rgbdDetector::IMAGE_WIDTH image_width;

    //Wrapper for key methods
    rgbdDetector rgbd_detector;

    //Non-maxima suppression neighbor size
    int nms_radius;

    //waittime
    int waittime;


public:
    linemod_detect(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold, float icp_maxCorresDist, uchar clustering_step,float orientation_clustering_th):
        it(nh),
        sub_color(nh,"/camera/rgb/image_rect_color",1),
        sub_depth(nh,"/camera/depth_registered/image_raw",1),
        depth_frame_id_("camera_link"),
        sync(SyncPolicy(1), sub_color, sub_depth),
        px_match_min_(0.25f),
        icp_dist_min_(0.06f),
        bias_x(56),
        image_width(rgbdDetector::ENSENSO),
        clustering_step_(clustering_step),
        waittime(1)
    {
        pc_rgb_pub_=nh.advertise<sensor_msgs::PointCloud2>("ensenso_rgb_pc",1);
        extract_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/render_pc",1);
        object_pose = nh.advertise<geometry_msgs::Transform>("object_pose",100);

        cout<<"templeate file name = "<<template_file_name<<endl;

        threshold=detect_score_threshold;

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
        K_rgb=Matx<double,3,3>(844.5966796875,0.0,338.907012939453125,
                               0.0,844.5966796875,232.793670654296875,
                               0.0,0.0,1.0);

        //Service client
        ensenso_registImg_client=nh.serviceClient<ensenso::RegistImage>("grab_registered_image");
        ensenso_singlePc_client=nh.serviceClient<ensenso::CaptureSinglePointCloud>("capture_single_point_cloud");

        //Orientation Clustering threshold
        orientation_clustering_th_=orientation_clustering_th;

    }

    virtual ~linemod_detect()
    {
        if(accumulator)
            free(accumulator);
        delete renderer_;
        delete renderer_iterator_;
    }

    void detect_cb(const sensor_msgs::Image& msg_rgb,sensor_msgs::PointCloud2 pc,bool is_rgb,vector<ClusterData>& cluster_datas)
    {
        //Publisher for visualize pointcloud in Rviz
        pointcloud_publisher scene_pc_pub(nh,string("/rgbDetect/scene"));
        pointcloud_publisher model_pc_pub(nh,string("/rgbDetect/pick_object"));
        pointcloud_publisher scene_cropped_pc_pub(nh,string("/rgbDetect/scene_roi"));

        //Convert image mgs to OpenCV
        Mat mat_rgb;
        Mat mat_grey;
        //if the image comes from monocular camera
        if(is_rgb)
        {
            cv_bridge::CvImagePtr img_ptr=cv_bridge::toCvCopy(msg_rgb,sensor_msgs::image_encodings::BGR8);
            img_ptr->image.copyTo(mat_rgb);
        }
        else //if the image comes from left camera of the stereo camera
        {
            cv_bridge::CvImagePtr img_ptr=cv_bridge::toCvCopy(msg_rgb,sensor_msgs::image_encodings::MONO8);
            img_ptr->image.copyTo(mat_grey);
            mat_rgb.create(mat_grey.rows,mat_grey.cols,CV_8UC3);
            int from_to[]={0,0,0,1,0,2};
            mixChannels(&mat_grey,1,&mat_rgb,1,from_to,3);

            //imshow("grey",mat_grey);
            imshow("conbined_gray",mat_rgb);
            waitKey(waittime);
        }

        //Convert pointcloud2 msg to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_median_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(pc,*pc_ptr);
        //Option: Median filter for PC smoothing
        //            pcl::MedianFilter<pcl::PointXYZ> median_filter;
        //            median_filter.setWindowSize(11);
        //            median_filter.setInputCloud(pc_ptr);
        //            median_filter.applyFilter(*pc_median_ptr);

        if(detector->classIds ().empty ())
        {
            ROS_INFO("Linemod detector is empty");
            return;
        }

        //Convert point cloud to depth image
        Mat mat_depth;
        pc2depth(pc_ptr,mat_depth);

        //Crop the image
        cv::Rect crop(bias_x,0,640,480);
        cv::GaussianBlur(mat_rgb, mat_rgb, cv::Size(3,3), 0, 0);
        Mat mat_rgb_crop=mat_rgb(crop);
        Mat initial_img=mat_rgb_crop.clone();
        Mat final=mat_rgb_crop.clone();
        Mat cluster_img=mat_rgb_crop.clone();
        Mat cluster_filter_img=mat_rgb_crop.clone();
        Mat nms_img=mat_rgb_crop.clone();

        //******************************************   1  linemod detection   ********************
        /*
           input: detector,sources,threshold
           output: matches
        */
        //Perform the detection
        std::vector<Mat> sources;
        sources.push_back (mat_rgb_crop);
        std::vector<linemod::Match> matches;
        double t;
        t=cv::getTickCount ();
        rgbd_detector.linemod_detection(detector,sources,threshold,matches);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by linemod detection: "<<t<<endl;

        //Display all the results
        // use contour points to show detect object   ----- it is one show method
        for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++){
            std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
            drawResponse(templates, 1, initial_img,cv::Point(it->x,it->y), 2);
        }
        // draw rectangle in the initial_img       ------  it is other show method
        for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++)
        {
            cv::Rect rect_tmp= Rects_[it->template_id];
            rect_tmp.x=it->x;
            rect_tmp.y=it->y;
            rectangle(initial_img,rect_tmp,Scalar(0,0,255),1);
        }
        Mat gray_initial_img,mat_gray_crop;
        cvtColor(mat_rgb_crop,mat_gray_crop,COLOR_BGR2GRAY);
        imshow("origin",mat_gray_crop);
        imshow("origin_rgb",mat_rgb_crop);
        imshow("initial",initial_img);
        cv::waitKey (waittime);


        //******************************************   2  Clustering based on Row,Col,Depth  ********************
        /*
           input: Obj_origin_dists, renderer_radius_min, clustering_step_, renderer_radius_step, matches
           output: map_match
        */
        //Clustering based on Row Col Depth
        std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
        rgbd_detector.rcd_voting(Obj_origin_dists,renderer_radius_min,clustering_step_,renderer_radius_step,matches,map_match);
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
        imshow("cluster",cluster_img);
        cv::waitKey (waittime);


        //******************************************   3  filter less than tresh map_match  ********************
        /*
           input: map_match,thresh
           output: map_match
        */
        //Filter based on size of clusters
        uchar thresh=2;
        rgbd_detector.cluster_filter(map_match,thresh);
        //Display all the results
        for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();it != map_match.end();it++)
        {
            Scalar color(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
            for(std::vector<linemod::Match>::iterator it_v= it->second.begin();it_v != it->second.end();it_v++)
                {
                cv::Rect rect_tmp= Rects_[it_v->template_id];
                rect_tmp.x=it_v->x;
                rect_tmp.y=it_v->y;
                rectangle(cluster_filter_img,rect_tmp,Scalar(0,0,255),2);
                }
        }
        imshow("cluster_filter",cluster_filter_img);
        cv::waitKey (waittime);


        //*******************************   4  create cluster_data and feed it cluster_average_score , index  ********************
        /*
           input: renderer_iterator_,K_tmp,Rs_,Ts_,map_match,depth_tmp,cluster_data
           output: cluster_data
        */
        //Compute criteria for each cluster
        //Output: Vecotor of ClusterData, each element of which contains index, score, flag of checking.
        vector<ClusterData> cluster_data;
        t=cv::getTickCount ();
        Matx33d K_tmp=Matx<double,3,3>(824.88983154296875, 0.0, 330.480316162109375,
                                       0.0, 824.88983154296875, 231.932891845703125,
                                       0.0, 0.0, 1.0);
        Mat depth_tmp=mat_depth.clone();
        rgbd_detector.cluster_scoring(renderer_iterator_,K_tmp,Rs_,Ts_,map_match,depth_tmp,cluster_data);
        //rgbd_detector.cluster_scoring(map_match,mat_depth,cluster_data);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by scroing: "<<t<<endl;


        //************   5  use nms_iou to erase overlap bbox(average rect), feed cluster_data use avg_rect,matches   **************
        /*
           input: cluster_data,nms_radius,Rects_,map_match
           output: cluster_data
        */
        //Non-maxima suppression
//        rgbd_detector.nonMaximaSuppression(cluster_data,nms_radius,Rects_,map_match);
        if(cluster_data.size()!=0)
        {
            rgbd_detector.nonMaximaSuppressionUsingIOU(cluster_data,nms_radius,Rects_,map_match);
             cout<<"cluster data size = "<<cluster_data.size()<<endl<<"cluster data [0] size = "<<cluster_data[0].matches.size()<<endl;
            for(int i=0;i<cluster_data.size();++i)
            {
              Scalar color_(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
              rectangle(nms_img,cluster_data[i].rect,color_,2);
            }
    //        for(vector<ClusterData>::iterator it_c = cluster_data.begin();it_c != cluster_data.end();++it_c)
    //        {
    //            vector<linemod::Match> matches_tmp = it_c->matches;
    //            for(std::vector<linemod::Match>::iterator it= matches_tmp.begin();it != matches_tmp.end();it++)
    //            {
    //                std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
    //                drawResponse(templates, 1, nms_img,cv::Point(it->x,it->y), 2);
    //            }
    //        }
    //        Mat gray_nms_img;
    //        cvtColor(nms_img,gray_nms_img,COLOR_BGR2GRAY);
            imshow("nms",nms_img);
            cv::waitKey (waittime);
        }


        //************   6  get Rough Pose By Clustering   **************
        /*
           input: cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists ......
           output: it->modle_pc, it->scene_pc, it->pose...
        */
        if(cluster_data.size()!=0)
        {
            //Pose average------------------------------------------------------------------------------------------
            t=cv::getTickCount ();
//            rgbd_detector.getRoughPoseByClusteringUsingNeigRect(cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
            rgbd_detector.getRoughPoseByClustering(cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose clustering: "<<t<<endl;


//            // trim scene pc use region growing and smooth ---------------------------------------------------------
//            t=cv::getTickCount ();
//            for(vector<ClusterData>::iterator it_c = cluster_data.begin();it_c != cluster_data.end();++it_c)
//            {
//                if(!it_c->scene_pc->empty())
//                {
//                    PointCloudXYZ::Ptr pc_reg_tmp1(new PointCloudXYZ);
//                    PointCloudXYZ::Ptr pc_reg_tmp2(new PointCloudXYZ);
//                    PointCloudXYZ::Ptr pc_reg_tmp3(new PointCloudXYZ);
////                    rgbd_detector.viz_pc(it_c->scene_pc);
//                    rgbd_detector.mls_smooth(it_c->scene_pc,0.01,pc_reg_tmp1);
//                    rgbd_detector.regionGrowingseg(it_c->scene_pc ,pc_reg_tmp2 ,6.0 ,0.05);     //origin is 7.0, 0.02
//                    rgbd_detector.mls_smooth(it_c->scene_pc,0.01,pc_reg_tmp3);
////                    rgbd_detector.viz_pc(it_c->scene_pc);
//                }

//            }
//            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
//            cout<<"Time consumed by smooth and regionGrowing: "<<t<<endl;

            //****************************   7  Pose refinement by icp   ************************************
            /*
              the last param is means is_viz , false means not viz;
              input: cluster_data,icp
              output: it->model, it->pose(update)
             */
            t=cv::getTickCount ();
            rgbd_detector.icpPoseRefine(cluster_data,icp,pc_ptr,image_width,bias_x,false);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose refinement: "<<t<<endl;
        }


//        t=cv::getTickCount ();
//        for(int i=0;i<cluster_data.size();++i)
//        {
//              rgbd_detector.templateRefinementEnsenso(cluster_data[i],pc_ptr,icp,renderer_iterator_);
//        }
//        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
//        cout<<"Time consumed by templateRefinement: "<<t<<endl;

        //Viz corresponding scene pc
//        for(int i=0;i<cluster_data.size();++i)
//            {
//            pcl::visualization::PCLVisualizer::Ptr v(new pcl::visualization::PCLVisualizer("view"));
//            v->setBackgroundColor(255,255,255);
//            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cluster_data[i].scene_pc,0,0,0);
//            v->addPointCloud(cluster_data[i].scene_pc,color,"cloud");
//            v->spin();
//        }


        //Hypothesis verification
//        rgbd_detector.hypothesisVerification(cluster_data,0.004,0.20,false);

        //Display all the bounding box
//        scene_pc_pub.publish (pc_ptr);
//        for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
//        {
//            //Display all grasping pose
//            Eigen::Affine3d grasp_pose;
////            rgbd_detector.graspingPoseBasedOnRegionGrowing (it->scene_pc,0.005,grasp_pose);
//            graspingPoseBasedOnRegionGrowing (it->scene_pc,0.005,grasp_pose);

//            tf::Transform grasp_pose_tf_viz;
//            tf::poseEigenToTF(grasp_pose,grasp_pose_tf_viz);
//            tf::TransformBroadcaster tf_broadcaster;
//            tf_broadcaster.sendTransform (tf::StampedTransform(grasp_pose_tf_viz,ros::Time::now(),"camera_link","grasp_frame"));

//            rectangle(display,it->rect,Scalar(0,0,255),2);
//            model_pc_pub.publish (it->model_pc,it->pose,cv::Scalar(255,0,0));
//            scene_cropped_pc_pub.publish(it->scene_pc,it->pose,cv::Scalar(0,255,0));
//            imshow("display",display);
//            cv::waitKey (0);
//        }

        //change pose direction
        for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
        {
            Eigen::Matrix3d R_temp =it->pose.linear();
            if(R_temp(0,0) < 0)  //x aixs is always parallel show coordinate x aixs
              {
                 R_temp.col(0) = -R_temp.col(0);
                 if(R_temp(1,1)>0)
                     R_temp.col(1) = -R_temp.col(1);
                 else
                     R_temp.col(2) = -R_temp.col(2);
              }
            else
            {
                if(R_temp(1,1)>0)
                {
                    R_temp.col(1) = -R_temp.col(1);
                    R_temp.col(2) = -R_temp.col(2);
                }
            }

            it->pose.linear() = R_temp ;
        }


        for(int i=0;i<cluster_data.size();++i)
        {
          rectangle(final,cluster_data[i].rect,Scalar(0,0,255),2);
        }
        imshow("display",final);
        cv::waitKey (waittime);

        //Viz in point cloud
//        vizResultPclViewer(cluster_data,pc_ptr);

//        delete renderer_;
//        delete renderer_iterator_;

        cluster_datas = cluster_data;


    }

    void graspingPoseBasedOnRegionGrowing(PointCloudXYZ::Ptr scene_pc,double offset,Eigen::Affine3d& grasping_pose)
    {
        //Params
        float k_neigh_norest=10;
        float k_neigh_reg=30;
        grasping_pose=Eigen::Affine3d::Identity ();

        //Denoise
        //rgbd_detector.statisticalOutlierRemoval(scene_pc,50,1.0);

        pcl::visualization::PCLVisualizer view("before region growing");
        view.addPointCloud(scene_pc,"before region growing");
        view.spin();

        //Smooth
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_(new pcl::search::KdTree<pcl::PointXYZ>);
        PointCloudXYZ::Ptr tmp_pc(new PointCloudXYZ);
        pcl::MovingLeastSquares<pcl::PointXYZ,pcl::PointXYZ> mls;
        mls.setInputCloud(scene_pc);
        mls.setComputeNormals(false);
        mls.setPolynomialFit(true);
        mls.setSearchMethod(tree_);
        mls.setSearchRadius(0.01);
        mls.process(*tmp_pc);
        pcl::copyPointCloud(*tmp_pc,*scene_pc);

        //Normal estimation
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
        norm_est.setKSearch (k_neigh_norest);
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
        norm_est.setInputCloud (scene_pc);
        norm_est.compute (*scene_normals);

        //Region growing
        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
        reg.setMinClusterSize (50);
        reg.setMaxClusterSize (1000000);
        pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
        reg.setSearchMethod (tree);
        reg.setNumberOfNeighbours (k_neigh_reg);
        reg.setInputCloud (scene_pc);
        //reg.setIndices (indices);
        reg.setInputNormals (scene_normals);
        /*The following two lines are most important part in the algorithm initialization, because they are responsible for the mentioned smoothness constraint.
         * First method sets the angle in radians that will be used as the allowable range for the normals deviation.
         * If the deviation between points normals is less than smoothness threshold then they are suggested to be in the same cluster (new point - the tested one - will be added to the cluster).
         * The second one is responsible for curvature threshold.
         * If two points have a small normals deviation then the disparity between their curvatures is tested. And if this value is less than curvature threshold then the algorithm will continue the growth of the cluster using new added point.*/
        reg.setSmoothnessThreshold (2.0 / 180.0 * M_PI);
        reg.setCurvatureThreshold (1.0);
        std::vector <pcl::PointIndices> clusters;
        reg.extract (clusters);
        std::sort(clusters.begin(),clusters.end(),sortRegionIndx);
        PointCloudXYZ::Ptr segmented_pc(new PointCloudXYZ);
        rgbd_detector.extractPointsByIndices (boost::make_shared<pcl::PointIndices>(clusters[0]),scene_pc,segmented_pc,false,false);

        pcl::visualization::PCLVisualizer view2("after region growing");
        view2.addPointCloud(segmented_pc,"after region growing");
        view2.spin();

        //search centroid
        //Get scene points centroid
        Eigen::Matrix<double,4,1> centroid;
        pcl::PointXYZ scene_surface_centroid;
        pcl::Normal scene_surface_normal;
        bool is_point_valid=true;
        if(pcl::compute3DCentroid<pcl::PointXYZ,double>(*segmented_pc,centroid)!=0)
        {
            //Search nearest point to centroid point
            int K = 1;
            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            float octree_res=0.001;
            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(octree_res));

            octree->setInputCloud(segmented_pc);
            octree->addPointsFromInputCloud();
            if (octree->nearestKSearch (pcl::PointXYZ(centroid[0],centroid[1],centroid[2]), K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                scene_surface_centroid=segmented_pc->points[pointIdxNKNSearch[0]];
                scene_surface_normal=scene_normals->at(clusters[0].indices[pointIdxNKNSearch[0]]);
            }
        }

        //Orientation
        Eigen::Vector4f ee_z_axis(0.0,0.0,1.0,0.0); //Just use camera's z axis. it could be robot's end-effector.
        Eigen::Vector4f obj_surface_normal(scene_surface_normal.normal_x,scene_surface_normal.normal_y,scene_surface_normal.normal_z,0.0);
        Eigen::Vector3d rot_axis(ee_z_axis[1]*obj_surface_normal[2]-ee_z_axis[2]*obj_surface_normal[1],
                ee_z_axis[2]*obj_surface_normal[0]-ee_z_axis[0]*obj_surface_normal[2],
                ee_z_axis[0]*obj_surface_normal[1]-ee_z_axis[1]*obj_surface_normal[0]);
        double rot_angle = M_PI-pcl::getAngle3D(ee_z_axis,obj_surface_normal);
        grasping_pose*=Eigen::AngleAxisd(-rot_angle,rot_axis);

        //Position
        grasping_pose.translation ()<<scene_surface_centroid.x,scene_surface_centroid.y,scene_surface_centroid.z;

        //Offset
        grasping_pose.translation()[0]-=offset*obj_surface_normal[0];
        grasping_pose.translation()[1]-=offset*obj_surface_normal[1];
        grasping_pose.translation()[2]-=offset*obj_surface_normal[2];


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
                //std::cout<<K_matrix<<std::endl;
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

    bool pc2depth(PointCloudXYZ::ConstPtr pc,Mat& mat_depth)
    {
        //Convert point cloud to depth image
        Mat mat_depth_m;
        CV_Assert(pc->empty() == false);
        if(!pc->empty()){
            int height=pc->height;
            int width=pc->width;
            mat_depth_m.create(height,width,CV_32FC1);
            for(int i=0;i<height;i++)
                for(int j=0;j<width;j++)
                    mat_depth_m.at<float>(i,j)=pc->at(j,i).z;
            //imshow("depth",mat_depth_m);
            //waitKey(0);
            //Convert m to mm
            mat_depth_m.convertTo(mat_depth,CV_16UC1,1000.0);
            return true;
        }else{
            ROS_ERROR("Empty pointcloud! Detection aborting!");
            return false;
        }
    }

    void vizResultPclViewer(const vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc_ptr)
    {
        pcl::visualization::PCLVisualizer view("v");
        view.addPointCloud(pc_ptr,"scene");
        for(int ii=0;ii<cluster_data.size();++ii)
        {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cluster_data[ii].model_pc,255,0,0);
            string str="model ";
            stringstream ss;
            ss<<ii;
            str+=ss.str();
            view.addPointCloud(cluster_data[ii].model_pc,color,str);

            //Get eigen tf
            //                Eigen::Affine3d obj_pose;
            //                Eigen::Matrix3d rot;
            //                cv2eigen(cluster_data[ii].orientation,rot);
            //                obj_pose.linear()=rot;
            //                obj_pose.translation()<<cluster_data[ii].position[0],cluster_data[ii].position[1],cluster_data[ii].position[2];

            Eigen::Affine3f obj_pose_f=cluster_data[ii].pose.cast<float>();
            //Eigen::Affine3f obj_pose_f=obj_pose.cast<float>();
            view.addCoordinateSystem(0.08,obj_pose_f);
            view.addCoordinateSystem(0.08);
        }
        view.spin();
    }

    ros::NodeHandle& getNodeHandle()
    {
        return nh;
    }

    void setNonMaximumSuppressionRadisu(double radius)\
    {
        nms_radius=radius;
    }

    void getImages(ensenso::RegistImage& srv)
    {
        ensenso_registImg_client.call(srv);
        int p=0;

    }

    //use h s v to discriminate color ,but just use one point
    void colorFilter2(cv::Mat inputImage ,vector<ClusterData>& cluster_data_colorFilter)
    {
//            Mat hsvImage;
//            cvtColor(inputImage, hsvImage, CV_BGR2HSV);
        vector<ClusterData> colorFilter_cluster_data;
        vector<ClusterData>::iterator it1 = cluster_data_colorFilter.begin();
        for(;it1 != cluster_data_colorFilter.end(); ++it1)
        {
            int i = it1->rect.y + it1->rect.height/2;
            int j = it1->rect.x + it1->rect.width/2;
            Mat hsvImage(1,1,CV_8UC3,Scalar(inputImage.at<Vec3b>(i,j)[0], inputImage.at<Vec3b>(i,j)[1], inputImage.at<Vec3b>(i,j)[2]));
            cvtColor(hsvImage, hsvImage, CV_BGR2HSV);
            cv::Vec3b inputColor = cv::Vec3b(hsvImage.at<Vec3b>(0,0)[0], hsvImage.at<Vec3b>(0,0)[1], hsvImage.at<Vec3b>(0,0)[2]);
            if( (inputColor[0]>0 && inputColor[0]<170) && (inputColor[1]<200 ) && (inputColor[2]<170 ))  // threshold value comes from getHSVHist.cpp in test_pcl_cv file
            {                                                                                            // change h s v value
                colorFilter_cluster_data.push_back(*it1);
            }
        }
        cluster_data_colorFilter.clear();
        cluster_data_colorFilter = colorFilter_cluster_data;

    }

    //calculate absolute rect
    void calAbsoluteRectangle(Mat src ,Rect& absoluteRect)
    {
        double t=cv::getTickCount ();
        imshow("1",src);
        Mat tmp,dst,dst_;

        cvtColor(src,tmp,COLOR_RGB2GRAY);
        blur(tmp,tmp,Size(3,3));
        cv::threshold(tmp,dst,60,255,CV_THRESH_BINARY_INV); //origin was 50 used to memortchip
        imshow("2",dst);
        dst_ = dst.clone();
        cvtColor(dst_,dst_,COLOR_GRAY2BGR);

        vector< vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(dst,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
    //    imshow("2_1",dst);

        std::sort(contours.begin(),contours.end(),sortPointSize);
        vector< vector<Point> > contours_;
        contours_.push_back(contours[0]);
        contours.clear();
        contours = contours_;

        Rect rect = boundingRect(Mat(contours[0]));
        rectangle(src,rect,Scalar(0,0,255),1,8);
        rectangle(dst_,rect,Scalar(0,0,255),1,8);
        absoluteRect = rect;

        imshow("3",src);
//        imshow("3_1",dst_);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by rect : "<<t<<endl;
        waitKey(waittime);
    }




};


class linemod_detect1
{
    ros::NodeHandle nh1;
    image_transport::ImageTransport it;
    ros::Subscriber sub_cam_info;
    //image_transport::Subscriber sub_color_;
    //image_transport::Subscriber sub_depth;
    //image_transport::Publisher pub_color_;
    //image_transport::Publisher pub_depth;
    message_filters::Subscriber<sensor_msgs::Image> sub_color;
    message_filters::Subscriber<sensor_msgs::Image> sub_depth;
//    message_filters::Synchronizer<SyncPolicy> sync;
//    ros::Publisher pc_rgb_pub_;
//    ros::Publisher extract_pc_pub;
//    ros::Publisher object_pose;

    //voting space
    unsigned int* accumulator;
    uchar clustering_step_;

public:
    Ptr<linemod::Detector> detector;
    float threshold;
    bool is_K_read;
    cv::Vec3f T_ref;

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
    Renderer3d *renderer1_;
    RendererIterator *renderer_iterator1_;

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
    ros::ServiceClient ensenso_registImg_client;
    ros::ServiceClient ensenso_singlePc_client;

    //Offset for cropped image
    int bias_x;
    rgbdDetector::IMAGE_WIDTH image_width;

    //Wrapper for key methods
    rgbdDetector rgbd_detector1;

    //Non-maxima suppression neighbor size
    int nms_radius;

    //waittime
    int waittime;


public:
    linemod_detect1(std::string template_file_name,std::string renderer_params_name,std::string mesh_path,float detect_score_threshold,int icp_max_iter,float icp_tr_epsilon,float icp_fitness_threshold, float icp_maxCorresDist, uchar clustering_step,float orientation_clustering_th):
        it(nh1),
        depth_frame_id_("camera_link"),
        px_match_min_(0.25f),
        icp_dist_min_(0.06f),
        bias_x(56),
        image_width(rgbdDetector::ENSENSO),
        clustering_step_(clustering_step),
        waittime(1)
    {
//        pc_rgb_pub_=nh.advertise<sensor_msgs::PointCloud2>("ensenso_rgb_pc",1);
//        extract_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/render_pc",1);
//        object_pose = nh.advertise<geometry_msgs::Transform>("object_pose",100);


        threshold=detect_score_threshold;

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
        renderer1_ = new Renderer3d(mesh_path);
        renderer1_->set_parameters(renderer_width, renderer_height, renderer_focal_length_x, renderer_focal_length_y, renderer_near, renderer_far);
        renderer_iterator1_ = new RendererIterator(renderer1_, renderer_n_points);
        renderer_iterator1_->angle_step_ = renderer_angle_step;
        renderer_iterator1_->radius_min_ = float(renderer_radius_min);
        renderer_iterator1_->radius_max_ = float(renderer_radius_max);
        renderer_iterator1_->radius_step_ = float(renderer_radius_step);

        icp.setMaximumIterations (icp_max_iter);
        icp.setMaxCorrespondenceDistance (icp_maxCorresDist);
        icp.setTransformationEpsilon (icp_tr_epsilon);
        icp.setEuclideanFitnessEpsilon (icp_fitness_threshold);


        R_diag=Matx<float,3,3>(1.0,0.0,0.0,
                               0.0,1.0,0.0,
                               0.0,0.0,1.0);
        K_rgb=Matx<double,3,3>(844.5966796875,0.0,338.907012939453125,
                               0.0,844.5966796875,232.793670654296875,
                               0.0,0.0,1.0);

//        //Service client
        ensenso_registImg_client=nh1.serviceClient<ensenso::RegistImage>("grab_registered_image");
        ensenso_singlePc_client=nh1.serviceClient<ensenso::CaptureSinglePointCloud>("capture_single_point_cloud");

        //Orientation Clustering threshold
        orientation_clustering_th_=orientation_clustering_th;

    }

    virtual ~linemod_detect1()
    {
        if(accumulator)
            free(accumulator);
        delete renderer1_;
        delete renderer_iterator1_;
    }

    void detect_cb(const sensor_msgs::Image& msg_rgb,sensor_msgs::PointCloud2 pc,bool is_rgb,vector<ClusterData>& cluster_datas)
    {
//        //Publisher for visualize pointcloud in Rviz
//        pointcloud_publisher scene_pc_pub(nh,string("/rgbDetect/scene"));
//        pointcloud_publisher model_pc_pub(nh,string("/rgbDetect/pick_object"));
//        pointcloud_publisher scene_cropped_pc_pub(nh,string("/rgbDetect/scene_roi"));

        //Convert image mgs to OpenCV
        Mat mat_rgb;
        Mat mat_grey;
        //if the image comes from monocular camera
        if(is_rgb)
        {
            cv_bridge::CvImagePtr img_ptr=cv_bridge::toCvCopy(msg_rgb,sensor_msgs::image_encodings::BGR8);
            img_ptr->image.copyTo(mat_rgb);
        }
        else //if the image comes from left camera of the stereo camera
        {
            cv_bridge::CvImagePtr img_ptr=cv_bridge::toCvCopy(msg_rgb,sensor_msgs::image_encodings::MONO8);
            img_ptr->image.copyTo(mat_grey);
            mat_rgb.create(mat_grey.rows,mat_grey.cols,CV_8UC3);
            int from_to[]={0,0,0,1,0,2};
            mixChannels(&mat_grey,1,&mat_rgb,1,from_to,3);

            //imshow("grey",mat_grey);
            imshow("conbined_gray",mat_rgb);
            waitKey(waittime);
        }

        //Convert pointcloud2 msg to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_median_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(pc,*pc_ptr);
        //Option: Median filter for PC smoothing
        //            pcl::MedianFilter<pcl::PointXYZ> median_filter;
        //            median_filter.setWindowSize(11);
        //            median_filter.setInputCloud(pc_ptr);
        //            median_filter.applyFilter(*pc_median_ptr);

        if(detector->classIds ().empty ())
        {
            ROS_INFO("Linemod detector is empty");
            return;
        }

        //Convert point cloud to depth image
        Mat mat_depth;
        pc2depth(pc_ptr,mat_depth);

        //Crop the image
        cv::Rect crop(bias_x,0,640,480);
        cv::GaussianBlur(mat_rgb, mat_rgb, cv::Size(3,3), 0, 0);
        Mat mat_rgb_crop=mat_rgb(crop);
        Mat initial_img=mat_rgb_crop.clone();
        Mat final=mat_rgb_crop.clone();
        Mat cluster_img=mat_rgb_crop.clone();
        Mat cluster_filter_img=mat_rgb_crop.clone();
        Mat nms_img=mat_rgb_crop.clone();

        //******************************************   1  linemod detection   ********************
        /*
           input: detector,sources,threshold
           output: matches
        */
        //Perform the detection
        std::vector<Mat> sources;
        sources.push_back (mat_rgb_crop);
        std::vector<linemod::Match> matches;
        double t;
        t=cv::getTickCount ();
        rgbd_detector1.linemod_detection(detector,sources,threshold,matches);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by linemod detection: "<<t<<endl;

        //Display all the results
        // use contour points to show detect object   ----- it is one show method
        for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++){
            std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
            drawResponse(templates, 1, initial_img,cv::Point(it->x,it->y), 2);
        }
        // draw rectangle in the initial_img       ------  it is other show method
        for(std::vector<linemod::Match>::iterator it= matches.begin();it != matches.end();it++)
        {
            cv::Rect rect_tmp= Rects_[it->template_id];
            rect_tmp.x=it->x;
            rect_tmp.y=it->y;
            rectangle(initial_img,rect_tmp,Scalar(0,0,255),1);
        }
        Mat gray_initial_img,mat_gray_crop;
        cvtColor(mat_rgb_crop,mat_gray_crop,COLOR_BGR2GRAY);
        imshow("origin",mat_gray_crop);
        imshow("origin_rgb",mat_rgb_crop);
        imshow("initial",initial_img);
        cv::waitKey (waittime);


        //******************************************   2  Clustering based on Row,Col,Depth  ********************
        /*
           input: Obj_origin_dists, renderer_radius_min, clustering_step_, renderer_radius_step, matches
           output: map_match
        */
        //Clustering based on Row Col Depth
        std::map<std::vector<int>, std::vector<linemod::Match> > map_match;
        rgbd_detector1.rcd_voting(Obj_origin_dists,renderer_radius_min,clustering_step_,renderer_radius_step,matches,map_match);
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
        imshow("cluster",cluster_img);
        cv::waitKey (waittime);


        //******************************************   3  filter less than tresh map_match  ********************
        /*
           input: map_match,thresh
           output: map_match
        */
        //Filter based on size of clusters
        uchar thresh=2;
        rgbd_detector1.cluster_filter(map_match,thresh);
        //Display all the results
        for(std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();it != map_match.end();it++)
        {
            Scalar color(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
            for(std::vector<linemod::Match>::iterator it_v= it->second.begin();it_v != it->second.end();it_v++)
                {
                cv::Rect rect_tmp= Rects_[it_v->template_id];
                rect_tmp.x=it_v->x;
                rect_tmp.y=it_v->y;
                rectangle(cluster_filter_img,rect_tmp,Scalar(0,0,255),2);
                }
        }
        imshow("cluster_filter",cluster_filter_img);
        cv::waitKey (waittime);


        //*******************************   4  create cluster_data and feed it cluster_average_score , index  ********************
        /*
           input: renderer_iterator_,K_tmp,Rs_,Ts_,map_match,depth_tmp,cluster_data
           output: cluster_data
        */
        //Compute criteria for each cluster
        //Output: Vecotor of ClusterData, each element of which contains index, score, flag of checking.
        vector<ClusterData> cluster_data;
        t=cv::getTickCount ();
        Matx33d K_tmp=Matx<double,3,3>(824.88983154296875, 0.0, 330.480316162109375,
                                       0.0, 824.88983154296875, 231.932891845703125,
                                       0.0, 0.0, 1.0);
        Mat depth_tmp=mat_depth.clone();
        rgbd_detector1.cluster_scoring(renderer_iterator1_,K_tmp,Rs_,Ts_,map_match,depth_tmp,cluster_data);
        //rgbd_detector1.cluster_scoring(map_match,mat_depth,cluster_data);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by scroing: "<<t<<endl;


        //************   5  use nms_iou to erase overlap bbox(average rect), feed cluster_data use avg_rect,matches   **************
        /*
           input: cluster_data,nms_radius,Rects_,map_match
           output: cluster_data
        */
        //Non-maxima suppression
//        rgbd_detector1.nonMaximaSuppression(cluster_data,nms_radius,Rects_,map_match);
        if(cluster_data.size()!=0)
        {
            rgbd_detector1.nonMaximaSuppressionUsingIOU(cluster_data,nms_radius,Rects_,map_match);
             cout<<"cluster data size = "<<cluster_data.size()<<endl<<"cluster data [0] size = "<<cluster_data[0].matches.size()<<endl;
            for(int i=0;i<cluster_data.size();++i)
            {
              Scalar color_(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
              rectangle(nms_img,cluster_data[i].rect,color_,2);
            }
    //        for(vector<ClusterData>::iterator it_c = cluster_data.begin();it_c != cluster_data.end();++it_c)
    //        {
    //            vector<linemod::Match> matches_tmp = it_c->matches;
    //            for(std::vector<linemod::Match>::iterator it= matches_tmp.begin();it != matches_tmp.end();it++)
    //            {
    //                std::vector<cv::linemod::Template> templates=detector->getTemplates(it->class_id, it->template_id);
    //                drawResponse(templates, 1, nms_img,cv::Point(it->x,it->y), 2);
    //            }
    //        }
    //        Mat gray_nms_img;
    //        cvtColor(nms_img,gray_nms_img,COLOR_BGR2GRAY);
            imshow("nms",nms_img);
            cv::waitKey (waittime);
        }


        //************   6  get Rough Pose By Clustering   **************
        /*
           input: cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists ......
           output: it->modle_pc, it->scene_pc, it->pose...
        */
        if(cluster_data.size()!=0)
        {
            //Pose average------------------------------------------------------------------------------------------
            t=cv::getTickCount ();
//            rgbd_detector1.getRoughPoseByClusteringUsingNeigRect(cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
            rgbd_detector1.getRoughPoseByClustering(cluster_data,pc_ptr,Rs_,Ts_,Distances_,Obj_origin_dists,orientation_clustering_th_,renderer_iterator1_,renderer_focal_length_x,renderer_focal_length_y,image_width,bias_x);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose clustering: "<<t<<endl;


//            // trim scene pc use region growing and smooth ---------------------------------------------------------
//            t=cv::getTickCount ();
//            for(vector<ClusterData>::iterator it_c = cluster_data.begin();it_c != cluster_data.end();++it_c)
//            {
//                if(!it_c->scene_pc->empty())
//                {
//                    PointCloudXYZ::Ptr pc_reg_tmp1(new PointCloudXYZ);
//                    PointCloudXYZ::Ptr pc_reg_tmp2(new PointCloudXYZ);
//                    PointCloudXYZ::Ptr pc_reg_tmp3(new PointCloudXYZ);
////                    rgbd_detector.viz_pc(it_c->scene_pc);
//                    rgbd_detector.mls_smooth(it_c->scene_pc,0.01,pc_reg_tmp1);
//                    rgbd_detector.regionGrowingseg(it_c->scene_pc ,pc_reg_tmp2 ,6.0 ,0.05);     //origin is 7.0, 0.02
//                    rgbd_detector.mls_smooth(it_c->scene_pc,0.01,pc_reg_tmp3);
////                    rgbd_detector.viz_pc(it_c->scene_pc);
//                }

//            }
//            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
//            cout<<"Time consumed by smooth and regionGrowing: "<<t<<endl;

            //****************************   7  Pose refinement by icp   ************************************
            /*
              the last param is means is_viz , false means not viz;
              input: cluster_data,icp
              output: it->model, it->pose(update)
             */
            t=cv::getTickCount ();
            rgbd_detector1.icpPoseRefine(cluster_data,icp,pc_ptr,image_width,bias_x,false);
            t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
            cout<<"Time consumed by pose refinement: "<<t<<endl;
        }


//        t=cv::getTickCount ();
//        for(int i=0;i<cluster_data.size();++i)
//        {
//              rgbd_detector.templateRefinementEnsenso(cluster_data[i],pc_ptr,icp,renderer_iterator_);
//        }
//        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
//        cout<<"Time consumed by templateRefinement: "<<t<<endl;

        //Viz corresponding scene pc
//        for(int i=0;i<cluster_data.size();++i)
//            {
//            pcl::visualization::PCLVisualizer::Ptr v(new pcl::visualization::PCLVisualizer("view"));
//            v->setBackgroundColor(255,255,255);
//            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cluster_data[i].scene_pc,0,0,0);
//            v->addPointCloud(cluster_data[i].scene_pc,color,"cloud");
//            v->spin();
//        }


        //Hypothesis verification
//        rgbd_detector.hypothesisVerification(cluster_data,0.004,0.20,false);

        //Display all the bounding box
//        scene_pc_pub.publish (pc_ptr);
//        for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
//        {
//            //Display all grasping pose
//            Eigen::Affine3d grasp_pose;
////            rgbd_detector.graspingPoseBasedOnRegionGrowing (it->scene_pc,0.005,grasp_pose);
//            graspingPoseBasedOnRegionGrowing (it->scene_pc,0.005,grasp_pose);

//            tf::Transform grasp_pose_tf_viz;
//            tf::poseEigenToTF(grasp_pose,grasp_pose_tf_viz);
//            tf::TransformBroadcaster tf_broadcaster;
//            tf_broadcaster.sendTransform (tf::StampedTransform(grasp_pose_tf_viz,ros::Time::now(),"camera_link","grasp_frame"));

//            rectangle(display,it->rect,Scalar(0,0,255),2);
//            model_pc_pub.publish (it->model_pc,it->pose,cv::Scalar(255,0,0));
//            scene_cropped_pc_pub.publish(it->scene_pc,it->pose,cv::Scalar(0,255,0));
//            imshow("display",display);
//            cv::waitKey (0);
//        }

        //change pose direction
        for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
        {
            Eigen::Matrix3d R_temp =it->pose.linear();
            if(R_temp(0,0) < 0)  //x aixs is always parallel show coordinate x aixs
              {
                 R_temp.col(0) = -R_temp.col(0);
                 if(R_temp(1,1)>0)
                     R_temp.col(1) = -R_temp.col(1);
                 else
                     R_temp.col(2) = -R_temp.col(2);
              }
            else
            {
                if(R_temp(1,1)>0)
                {
                    R_temp.col(1) = -R_temp.col(1);
                    R_temp.col(2) = -R_temp.col(2);
                }
            }

            it->pose.linear() = R_temp ;
        }


        for(int i=0;i<cluster_data.size();++i)
        {
          rectangle(final,cluster_data[i].rect,Scalar(0,0,255),2);
        }
        imshow("display",final);
        cv::waitKey (waittime);

        //Viz in point cloud
//        vizResultPclViewer(cluster_data,pc_ptr, "vv");

        cluster_datas = cluster_data;

    }

    void graspingPoseBasedOnRegionGrowing(PointCloudXYZ::Ptr scene_pc,double offset,Eigen::Affine3d& grasping_pose)
    {
        //Params
        float k_neigh_norest=10;
        float k_neigh_reg=30;
        grasping_pose=Eigen::Affine3d::Identity ();

        //Denoise
        //rgbd_detector.statisticalOutlierRemoval(scene_pc,50,1.0);

        pcl::visualization::PCLVisualizer view("before region growing");
        view.addPointCloud(scene_pc,"before region growing");
        view.spin();

        //Smooth
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_(new pcl::search::KdTree<pcl::PointXYZ>);
        PointCloudXYZ::Ptr tmp_pc(new PointCloudXYZ);
        pcl::MovingLeastSquares<pcl::PointXYZ,pcl::PointXYZ> mls;
        mls.setInputCloud(scene_pc);
        mls.setComputeNormals(false);
        mls.setPolynomialFit(true);
        mls.setSearchMethod(tree_);
        mls.setSearchRadius(0.01);
        mls.process(*tmp_pc);
        pcl::copyPointCloud(*tmp_pc,*scene_pc);

        //Normal estimation
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
        norm_est.setKSearch (k_neigh_norest);
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
        norm_est.setInputCloud (scene_pc);
        norm_est.compute (*scene_normals);

        //Region growing
        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
        reg.setMinClusterSize (50);
        reg.setMaxClusterSize (1000000);
        pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
        reg.setSearchMethod (tree);
        reg.setNumberOfNeighbours (k_neigh_reg);
        reg.setInputCloud (scene_pc);
        //reg.setIndices (indices);
        reg.setInputNormals (scene_normals);
        /*The following two lines are most important part in the algorithm initialization, because they are responsible for the mentioned smoothness constraint.
         * First method sets the angle in radians that will be used as the allowable range for the normals deviation.
         * If the deviation between points normals is less than smoothness threshold then they are suggested to be in the same cluster (new point - the tested one - will be added to the cluster).
         * The second one is responsible for curvature threshold.
         * If two points have a small normals deviation then the disparity between their curvatures is tested. And if this value is less than curvature threshold then the algorithm will continue the growth of the cluster using new added point.*/
        reg.setSmoothnessThreshold (2.0 / 180.0 * M_PI);
        reg.setCurvatureThreshold (1.0);
        std::vector <pcl::PointIndices> clusters;
        reg.extract (clusters);
        std::sort(clusters.begin(),clusters.end(),sortRegionIndx);
        PointCloudXYZ::Ptr segmented_pc(new PointCloudXYZ);
        rgbd_detector1.extractPointsByIndices (boost::make_shared<pcl::PointIndices>(clusters[0]),scene_pc,segmented_pc,false,false);

        pcl::visualization::PCLVisualizer view2("after region growing");
        view2.addPointCloud(segmented_pc,"after region growing");
        view2.spin();

        //search centroid
        //Get scene points centroid
        Eigen::Matrix<double,4,1> centroid;
        pcl::PointXYZ scene_surface_centroid;
        pcl::Normal scene_surface_normal;
        bool is_point_valid=true;
        if(pcl::compute3DCentroid<pcl::PointXYZ,double>(*segmented_pc,centroid)!=0)
        {
            //Search nearest point to centroid point
            int K = 1;
            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            float octree_res=0.001;
            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(octree_res));

            octree->setInputCloud(segmented_pc);
            octree->addPointsFromInputCloud();
            if (octree->nearestKSearch (pcl::PointXYZ(centroid[0],centroid[1],centroid[2]), K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                scene_surface_centroid=segmented_pc->points[pointIdxNKNSearch[0]];
                scene_surface_normal=scene_normals->at(clusters[0].indices[pointIdxNKNSearch[0]]);
            }
        }

        //Orientation
        Eigen::Vector4f ee_z_axis(0.0,0.0,1.0,0.0); //Just use camera's z axis. it could be robot's end-effector.
        Eigen::Vector4f obj_surface_normal(scene_surface_normal.normal_x,scene_surface_normal.normal_y,scene_surface_normal.normal_z,0.0);
        Eigen::Vector3d rot_axis(ee_z_axis[1]*obj_surface_normal[2]-ee_z_axis[2]*obj_surface_normal[1],
                ee_z_axis[2]*obj_surface_normal[0]-ee_z_axis[0]*obj_surface_normal[2],
                ee_z_axis[0]*obj_surface_normal[1]-ee_z_axis[1]*obj_surface_normal[0]);
        double rot_angle = M_PI-pcl::getAngle3D(ee_z_axis,obj_surface_normal);
        grasping_pose*=Eigen::AngleAxisd(-rot_angle,rot_axis);

        //Position
        grasping_pose.translation ()<<scene_surface_centroid.x,scene_surface_centroid.y,scene_surface_centroid.z;

        //Offset
        grasping_pose.translation()[0]-=offset*obj_surface_normal[0];
        grasping_pose.translation()[1]-=offset*obj_surface_normal[1];
        grasping_pose.translation()[2]-=offset*obj_surface_normal[2];


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
                //std::cout<<K_matrix<<std::endl;
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

    bool pc2depth(PointCloudXYZ::ConstPtr pc,Mat& mat_depth)
    {
        //Convert point cloud to depth image
        Mat mat_depth_m;
        CV_Assert(pc->empty() == false);
        if(!pc->empty()){
            int height=pc->height;
            int width=pc->width;
            mat_depth_m.create(height,width,CV_32FC1);
            for(int i=0;i<height;i++)
                for(int j=0;j<width;j++)
                    mat_depth_m.at<float>(i,j)=pc->at(j,i).z;
            //imshow("depth",mat_depth_m);
            //waitKey(0);
            //Convert m to mm
            mat_depth_m.convertTo(mat_depth,CV_16UC1,1000.0);
            return true;
        }else{
            ROS_ERROR("Empty pointcloud! Detection aborting!");
            return false;
        }
    }

    void vizResultPclViewer(const vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc_ptr,string v_name="v")
    {
        pcl::visualization::PCLVisualizer view(v_name);
        view.addPointCloud(pc_ptr,"scene");
        for(int ii=0;ii<cluster_data.size();++ii)
        {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cluster_data[ii].model_pc,255,0,0);
            string str="model ";
            stringstream ss;
            ss<<ii;
            str+=ss.str();
            view.addPointCloud(cluster_data[ii].model_pc,color,str);

            //Get eigen tf
            //                Eigen::Affine3d obj_pose;
            //                Eigen::Matrix3d rot;
            //                cv2eigen(cluster_data[ii].orientation,rot);
            //                obj_pose.linear()=rot;
            //                obj_pose.translation()<<cluster_data[ii].position[0],cluster_data[ii].position[1],cluster_data[ii].position[2];

            Eigen::Affine3f obj_pose_f=cluster_data[ii].pose.cast<float>();
            //Eigen::Affine3f obj_pose_f=obj_pose.cast<float>();
            view.addCoordinateSystem(0.08,obj_pose_f);
            view.addCoordinateSystem(0.08);
        }
        view.spin();
    }

    ros::NodeHandle& getNodeHandle()
    {
        return nh1;
    }

    void setNonMaximumSuppressionRadisu(double radius)\
    {
        nms_radius=radius;
    }

    void getImages(ensenso::RegistImage& srv)
    {
        ensenso_registImg_client.call(srv);
        int p=0;

    }

    //use h s v to discriminate color ,but just use one point
    void colorFilter2(cv::Mat inputImage ,vector<ClusterData>& cluster_data_colorFilter)
    {
//            Mat hsvImage;
//            cvtColor(inputImage, hsvImage, CV_BGR2HSV);
        vector<ClusterData> colorFilter_cluster_data;
        vector<ClusterData>::iterator it1 = cluster_data_colorFilter.begin();
        for(;it1 != cluster_data_colorFilter.end(); ++it1)
        {
            int i = it1->rect.y + it1->rect.height/2;
            int j = it1->rect.x + it1->rect.width/2;
            Mat hsvImage(1,1,CV_8UC3,Scalar(inputImage.at<Vec3b>(i,j)[0], inputImage.at<Vec3b>(i,j)[1], inputImage.at<Vec3b>(i,j)[2]));
            cvtColor(hsvImage, hsvImage, CV_BGR2HSV);
            cv::Vec3b inputColor = cv::Vec3b(hsvImage.at<Vec3b>(0,0)[0], hsvImage.at<Vec3b>(0,0)[1], hsvImage.at<Vec3b>(0,0)[2]);
            if( (inputColor[0]>0 && inputColor[0]<170) && (inputColor[1]<200 ) && (inputColor[2]<170 ))  // threshold value comes from getHSVHist.cpp in test_pcl_cv file
            {                                                                                            // change h s v value
                colorFilter_cluster_data.push_back(*it1);
            }
        }
        cluster_data_colorFilter.clear();
        cluster_data_colorFilter = colorFilter_cluster_data;

    }

    //calculate absolute rect
    void calAbsoluteRectangle(Mat src ,Rect& absoluteRect)
    {
        double t=cv::getTickCount ();
        imshow("1",src);
        Mat tmp,dst,dst_;

        cvtColor(src,tmp,COLOR_RGB2GRAY);
        blur(tmp,tmp,Size(3,3));
        cv::threshold(tmp,dst,60,255,CV_THRESH_BINARY_INV); //origin was 50 used to memortchip
        imshow("2",dst);
        dst_ = dst.clone();
        cvtColor(dst_,dst_,COLOR_GRAY2BGR);

        vector< vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(dst,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
    //    imshow("2_1",dst);

        std::sort(contours.begin(),contours.end(),sortPointSize);
        vector< vector<Point> > contours_;
        contours_.push_back(contours[0]);
        contours.clear();
        contours = contours_;

        Rect rect = boundingRect(Mat(contours[0]));
        rectangle(src,rect,Scalar(0,0,255),1,8);
        rectangle(dst_,rect,Scalar(0,0,255),1,8);
        absoluteRect = rect;

        imshow("3",src);
//        imshow("3_1",dst_);
        t=(cv::getTickCount ()-t)/cv::getTickFrequency ();
        cout<<"Time consumed by rect : "<<t<<endl;
        waitKey(waittime);
    }




};

Eigen::Affine3d getTool0toDepthTF(double x, double y,double z, double qw,double qx,double qy,double qz)
{
    Eigen::Affine3d pose_Tool0Tdep_;

    //Translation
    pose_Tool0Tdep_.translation()<< x,y,z;

    //Rotation
    Eigen::Quaterniond quat(qw,qx,qy,qz);
    pose_Tool0Tdep_.linear() = quat.toRotationMatrix();

    return pose_Tool0Tdep_;
}

Eigen::Affine3d tfbaseTotool0()
{
    //Get tf from BASE to TOOL0
    tf::TransformListener listener;
    tf::StampedTransform transform_stamped;
    ros::Time now(ros::Time::now());
    listener.waitForTransform("base","tool0",now,ros::Duration(1.5));
    listener.lookupTransform("base","tool0",ros::Time(0),transform_stamped);
    Eigen::Affine3d pose_baseTtool0;
    tf::poseTFToEigen(transform_stamped,pose_baseTtool0);

    return pose_baseTtool0;
}

geometry_msgs::Transform affineTotrans(Eigen::Affine3d input)
{
    geometry_msgs::Transform pose_tf;
    pose_tf.translation.x = input.translation()[0];   // - 0.025
    pose_tf.translation.y = input.translation()[1];         // +0.01
    pose_tf.translation.z = input.translation()[2];

    Eigen::Vector4d R_quat_vec(0.0,0.0,0.0,0.0);
    R_quat_vec=Eigen::Quaterniond(input.linear()).coeffs();  //qw,qx,qy,qz maybe error,so i use qx,qy,qz,qw
    pose_tf.rotation.x = R_quat_vec[0];                                     // yes, it is right;
    pose_tf.rotation.y = R_quat_vec[1];
    pose_tf.rotation.z = R_quat_vec[2];
    pose_tf.rotation.w = R_quat_vec[3];

    return pose_tf;
}


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
    int nms_neighbor_size;

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
    nms_neighbor_size=atoi(argv[11]);
    linemod_detect detector(linemod_template_path,renderer_param_path,model_stl_path,
                            detect_score_th,icp_max_iter,icp_tr_epsilon,icp_fitness_th,
                            icp_maxCorresDist,clustering_step,orientation_clustering_step);
    detector.setNonMaximumSuppressionRadisu(nms_neighbor_size);

    // do for cpu
    linemod_template_path=argv[12];
    renderer_param_path=argv[13];
    model_stl_path=argv[14];
    detect_score_th=atof(argv[15]);
    linemod_detect1 detector1(linemod_template_path,renderer_param_path,model_stl_path,
                              detect_score_th,icp_max_iter,icp_tr_epsilon,icp_fitness_th,
                              icp_maxCorresDist,clustering_step,orientation_clustering_step);
    detector1.setNonMaximumSuppressionRadisu(nms_neighbor_size);


    ensenso::RegistImage srv;
    srv.request.is_rgb=true;
    ros::Rate loop(1);

//    string img_path=argv[12];
//    string pc_path=argv[13];
//    ros::Time now =ros::Time::now();
//    Mat cv_img=imread(img_path,IMREAD_COLOR);
//    cv::imshow("ddd", cv_img);
//    cv::waitKey(0);
//    cv_bridge::CvImagePtr bridge_img_ptr(new cv_bridge::CvImage);
//    bridge_img_ptr->image=cv_img;
//    bridge_img_ptr->encoding="bgr8";
//    bridge_img_ptr->header.stamp=now;
//    srv.response.image = *bridge_img_ptr->toImageMsg();

//    PointCloudXYZ::Ptr pc(new PointCloudXYZ);
//    pcl::io::loadPCDFile(pc_path,*pc);
//    pcl::toROSMsg(*pc,srv.response.pointcloud);
//    srv.response.pointcloud.header.frame_id="/camera_link";
//    srv.response.pointcloud.header.stamp=now;

    // make a flage, to choose memory card or cpu
    bool flage = true;

    ros::NodeHandle nh2;
    ros::Publisher object_pose;
    object_pose = nh2.advertise<geometry_msgs::Transform>("object_pose",100);

    while(ros::ok())
    {
        vector<ClusterData> targets;

        if(flage)
        {
            detector.getImages(srv);
            double t_total =cv::getTickCount ();
            detector.detect_cb(srv.response.image,srv.response.pointcloud,srv.request.is_rgb,targets);
            t_total=(cv::getTickCount ()-t_total)/cv::getTickFrequency ();
            cout<<"Time consumed by total time: "<<t_total<<" s"<<endl<<endl;

            flage = false;
        }

        else
        {
            detector1.getImages(srv);
            double t_total1 =cv::getTickCount ();
            detector1.detect_cb(srv.response.image,srv.response.pointcloud,srv.request.is_rgb,targets);
            t_total1=(cv::getTickCount ()-t_total1)/cv::getTickFrequency ();
            cout<<"Time consumed by total time: "<<t_total1<<" s"<<endl<<endl;

            flage =  true;
        }

        if(targets.size() != 0)
        {
            for(vector<ClusterData>::iterator it_target=targets.begin();it_target!=targets.end();++it_target)
            {
                //tf from tool0 to depth camera; it is mean eye in hand result
                Eigen::Affine3d pose_tool0Tdep;
                pose_tool0Tdep = getTool0toDepthTF(0.0672827, -0.0546864, 0.0466534, 0.701074, 2.999e-05, 0.00514592, 0.71307);

                //tf listern base to tool0
                Eigen::Affine3d pose_baseTtool0;
                pose_baseTtool0 = tfbaseTotool0();

                //tf Template matching computation depth to obj
                Eigen::Affine3d pose_depTobj;
                pose_depTobj = it_target->pose;

                //tf calculate base to obj
                Eigen::Affine3d pose_baseTobj;
                pose_baseTobj = pose_baseTtool0 * pose_tool0Tdep * pose_depTobj;

                //change Affine3d to geometry_msgs::Transform for publish
                geometry_msgs::Transform pose_publish;
                pose_publish = affineTotrans(pose_baseTobj);

                object_pose.publish(pose_publish);

            }
        }







        loop.sleep();
    }

    ros::spin ();
}
