#ifndef RGBDDETECTOR_H
#define RGBDDETECTOR_H

//ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/String.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>

//std
#include <iostream>

//Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Core>


//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/core/eigen.hpp>

//pcl
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/median_filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/octree/octree.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/features/board.h>
#include <pcl/surface/mls.h>


//ork
#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;

using namespace std;
using namespace cv;

//Score based on evaluation for 1 cluster
struct ClusterData{

    ClusterData():
        model_pc(new PointCloudXYZ),
        dense_scene_pc(new PointCloudXYZ)
    {
        index.resize(3);
        is_checked=false;
        dist=0.0;

    }

    ClusterData(const vector<int> index_,double score_):
        index(index_),
        score(score_),
        is_checked(false),
        dist(0.0),
        model_pc(new PointCloudXYZ),
        dense_scene_pc(new PointCloudXYZ)
    {

    }

    vector<linemod::Match> matches;
    vector<int> index;
    double score;
    bool is_checked;
    Rect rect;
    cv::Matx33d orientation;
    cv::Vec3d position;
    cv::Vec3d T_match;
    cv::Mat mask;
    cv::Mat K_matrix;
    double dist;
    PointCloudXYZ::Ptr model_pc;
    PointCloudXYZ::Ptr scene_pc;
    PointCloudXYZ::Ptr dense_scene_pc;
    Eigen::Affine3d pose;
    Eigen::Affine3d pose_pC1;
    Eigen::Affine3d c1_tr_c2;
    Scalar color;

};

bool sortOrienCluster(const vector<Eigen::Matrix3d>& cluster1,const vector<Eigen::Matrix3d>& cluster2)
{
    return(cluster1.size()>cluster2.size());
}

bool sortIdCluster(const vector<int>& cluster1,const vector<int>& cluster2)
{
    return(cluster1.size()>cluster2.size());
}

bool sortXyCluster(const vector<pair<int,int> >& cluster1,const vector<pair<int,int> >& cluster2)
{
    return(cluster1.size()>cluster2.size());
}

bool sortScoreCluster(const ClusterData& cluster1,const ClusterData& cluster2)
{
    return(cluster1.score>cluster2.score);
}

bool sortRegionIndx(const pcl::PointIndices& indice1,const pcl::PointIndices& indice2)
{
    return(indice1.indices.size()>indice2.indices.size());
}

class rgbdDetector
{

public:
    enum IMAGE_WIDTH{
        ENSENSO = 752,
        CARMINE = 640
    };


public:
    rgbdDetector();

    void linemod_detection(Ptr<linemod::Detector> linemod_detector, const vector<Mat>& sources, const float& threshold, std::vector<linemod::Match> &matches);

    void rcd_voting(vector<double>& Obj_origin_dists,const double& renderer_radius_min,const int& vote_row_col_step,const double& renderer_radius_step_,const vector<linemod::Match>& matches,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match);

    void cluster_filter(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,int thresh);

    void cluster_filter(std::vector<ClusterData>& cluster_data,int thresh);

    void cluster_scoring(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,Mat& depth_img,std::vector<ClusterData>& cluster_data);

    void cluster_scoring(RendererIterator *renderer_iterator_,Matx33d& K_rgb,vector<Mat>& Rs_,vector<Mat>& Ts_,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,Mat& depth_img,std::vector<ClusterData>& cluster_data);

    double similarity_score_calc(std::vector<linemod::Match> match_cluster);

    double depth_normal_diff_calc(RendererIterator *renderer_iterator_,Matx33d& K_rgb,vector<Mat>& Rs_,vector<Mat>& Ts_,std::vector<linemod::Match> match_cluster, Mat& depth_img);

    double depth_diff(Mat& depth_img,Mat& depth_template,cv::Mat& template_mask,cv::Rect& rect);

    double normal_diff(cv::Mat& depth_img, Mat& depth_template, cv::Mat& template_mask, Rect &template_mask_rect, cv::Rect& rect, Matx33d &K_rgb);

    double getClusterScore(const double& depth_diff_score,const double& normal_diff_score);

    void nonMaximaSuppression(vector<ClusterData>& cluster_data,const double& neighborSize, vector<Rect>& Rects_,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match);

    void nonMaximaSuppressionUsingIOU(vector<ClusterData>& cluster_data,const double& neighborSize, vector<Rect>& Rects_,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match);

    float computeIoU(cv::Rect rect1, cv::Rect rect2);

    void getRoughPoseByClustering(vector<ClusterData>& cluster_data, PointCloudXYZ::Ptr pc, vector<Mat> &Rs_, vector<Mat> &Ts_, vector<double> &Distances_, vector<double> &Obj_origin_dists, float orientation_clustering_th_, RendererIterator *renderer_iterator_, double &renderer_focal_length_x, double& renderer_focal_length_y, IMAGE_WIDTH &image_width, int& bias_x);

    bool orientationCompare(Eigen::Matrix3d& orien1,Eigen::Matrix3d& orien2,double thresh);

    pcl::PointIndices::Ptr getPointCloudIndices(vector<ClusterData>::iterator& it, IMAGE_WIDTH image_width, int bias_x, Rect mask_rect);

    pcl::PointIndices::Ptr getPointCloudIndices(const cv::Rect& rect, IMAGE_WIDTH image_width,int bias_x);

    void extractPointsByIndices(pcl::PointIndices::Ptr indices, const PointCloudXYZ::Ptr ref_pts, PointCloudXYZ::Ptr extracted_pts, bool is_negative,bool is_organised);

    void icpPoseRefine(vector<ClusterData>& cluster_data, pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> &icp, PointCloudXYZ::Ptr pc, IMAGE_WIDTH image_width, int bias_x, bool is_viz);

    void icpNonLinearPoseRefine(vector<ClusterData>& cluster_data, PointCloudXYZ::Ptr pc, IMAGE_WIDTH image_width, int bias_x, Rect mask_rect);

    void euclidianClustering(PointCloudXYZ::Ptr pts,float dist);

    void statisticalOutlierRemoval(PointCloudXYZ::Ptr pts, int num_neighbor,float stdDevMulThresh);

    void voxelGridFilter(PointCloudXYZ::Ptr pts, float leaf_size);

    void hypothesisVerification(vector<ClusterData>& cluster_data, float octree_res, float thresh, bool is_viz);

    void getPositionByDistanceOffset(vector<ClusterData>::iterator it, PointCloudXYZ::Ptr pc, int x_index, int y_index, IMAGE_WIDTH image_width, int bias_x, double DistanceOffset, double DistanceFromCamToObj, bool is_center_hole, Eigen::Vector3d& position, Eigen::Affine3d& transform, Rect mask_rect);

    void getPositionByROICenter(vector<ClusterData>::iterator it,PointCloudXYZ::Ptr pc,int x_index_scene,int y_index_scene,double DistanceFromCamToObj,Eigen::Affine3d& transform,Eigen::Vector3d& position);

    void getPositionBySurfaceCentroid(PointCloudXYZ::Ptr scene_pc,PointCloudXYZ::Ptr model_pc,double DistanceFromCamToObj,Eigen::Vector3d& position,Eigen::Affine3d& transform);

    void getPoseByLocalDescriptor(PointCloudXYZ::Ptr scene_pc, PointCloudXYZ::Ptr model_pc, double DistanceFromCamToObj,Eigen::Affine3d& pose, Eigen::Vector3d &position);

    void graspingPoseBasedOnRegionGrowing(PointCloudXYZ::Ptr dense_scene_pc, PointCloudXYZ::Ptr scene_pc, float normal_thresh, float curvature_thresh, double offset, Eigen::Affine3d& grasping_pose, bool is_viz);

    //Utilities
    cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename);
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
                                   double& renderer_far);


};

class pointcloud_publisher
{
public:
    ros::Publisher publisher;
    sensor_msgs::PointCloud2 pc_msg;
    tf::TransformBroadcaster tf_broadcaster;
public:
    pointcloud_publisher(ros::NodeHandle& nh,const string& topic);
    void publish(PointCloudXYZ::Ptr pc);
    void publish(sensor_msgs::PointCloud2& pc_msg);
    void publish(PointCloudXYZ::Ptr pc, Eigen::Affine3d pose, const Scalar& color);
};

#endif // RGBDDETECTOR_H


