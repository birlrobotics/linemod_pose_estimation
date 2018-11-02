#include <linemod_pose_estimation/rgbdDetector.h>
//#include <opencv2/opencv.hpp>
//#include <opencv2/rgbd/rgbd.hpp>
//#include <opencv2/core/eigen.hpp>
//#include </opt/ros/indigo/include/opencv2/rgbd/rgbd.hpp>

using namespace cv;

//Constructor
rgbdDetector::rgbdDetector()
{
    //Read all linemod-related params
    //linemod_detector=readLinemod (linemod_template_path);
//    readLinemodTemplateParams (linemod_render_path,Rs_,Ts_,Distances_,Obj_origin_dists,
//                               Ks_,Rects_,renderer_n_points,renderer_angle_step,
//                               renderer_radius_min,renderer_radius_max,
//                               renderer_radius_step,renderer_width,renderer_height,
//                               renderer_focal_length_x,renderer_focal_length_y,
//                               renderer_near,renderer_far);

//    //Initiate ICP
//    icp.setMaximumIterations (icp_max_iter);
//    icp.setMaxCorrespondenceDistance (icp_maxCorresDist);
//    icp.setTransformationEpsilon (icp_tr_epsilon);
//    icp.setEuclideanFitnessEpsilon (icp_fitness_threshold);

//    linemod_thresh=detect_score_threshold;
}

//Perform the LINEMOD detection
void rgbdDetector::linemod_detection(Ptr<linemod::Detector> linemod_detector,const vector<Mat>& sources,const float& threshold,std::vector<linemod::Match>& matches)
{
    linemod_detector->match (sources,threshold,matches,std::vector<String>(),noArray());
}

void rgbdDetector::rcd_voting(vector<double>& Obj_origin_dists,const double& renderer_radius_min,const int& vote_row_col_step,const double& renderer_radius_step_,const vector<linemod::Match>& matches,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match)
{
    //----------------3D Voting-----------------------------------------------------//

    int voting_width_step=vote_row_col_step; //Unit: pixel
    int voting_height_step=vote_row_col_step; //Unit: pixel, width step and height step suppose to be the same
    float voting_depth_step=renderer_radius_step_;//Unit: m

    BOOST_FOREACH(const linemod::Match& match,matches){
        //Get height(row), width(cols), depth index
        int height_index=match.y/voting_height_step;
        int width_index=match.x/voting_width_step;
        float depth = Obj_origin_dists[match.template_id];//the distance from the object origin to the camera origin
        int depth_index=(int)((depth-renderer_radius_min)/voting_depth_step);

        //fill the index
        vector<int> index(3);
        index[0]=height_index;
        index[1]=width_index;
        index[2]=depth_index;

        //Use MAP to store matches of the same index
        if(map_match.find (index)==map_match.end ())
        {
            std::vector<linemod::Match> temp;
            temp.push_back (match);
            map_match.insert(pair<std::vector<int>,std::vector<linemod::Match> >(index,temp));
        }
        else
        {
            map_match[index].push_back(match);
        }

    }
}

void rgbdDetector::cluster_filter(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,int thresh)
{
    //assert(map_match.size() != 0);
    std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it= map_match.begin();
    for(;it!=map_match.end();++it)
    {
       if(it->second.size()<=thresh)
       {
           map_match.erase(it);
       }
    }


}

void rgbdDetector::cluster_filter(std::vector<ClusterData>& cluster_data,int thresh)
{
    std::vector<ClusterData>::iterator it = cluster_data.begin();
    for(;it!=cluster_data.end();)
    {
        if(it->matches.size()<thresh)
        {
            it=cluster_data.erase(it);
        }else
        {
            it++;
        }
    }


}

void rgbdDetector::cluster_scoring(std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,Mat& depth_img,std::vector<ClusterData>& cluster_data)
{
    std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it_map= map_match.begin();
    for(;it_map!=map_match.end();++it_map)
    {
        //Perform depth difference computation and normal difference computation.The process is not optimized so it will be slow when a cluster contains two many templates.
        //double score=depth_normal_diff_calc(it_map->second,depth_img);
        //Options: similairy score computation: averaging templates' similarity score.
        double score=similarity_score_calc(it_map->second);
        cluster_data.push_back(ClusterData(it_map->first,score));

    }
}

void rgbdDetector::cluster_scoring(RendererIterator *renderer_iterator_,Matx33d& K_rgb,vector<Mat>& Rs_,vector<Mat>& Ts_,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match,Mat& depth_img,std::vector<ClusterData>& cluster_data)
{
    std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it_map= map_match.begin();
    for(;it_map!=map_match.end();++it_map)
    {
        //Perform depth difference computation and normal difference computation
        //double score=depth_normal_diff_calc(renderer_iterator_,K_rgb,Rs_,Ts_,it_map->second,depth_img);
        //Options: similairy score computation
        double score=similarity_score_calc(it_map->second);
        cluster_data.push_back(ClusterData(it_map->first,score));

    }
}

double rgbdDetector::similarity_score_calc(std::vector<linemod::Match> match_cluster)
{
    double sum_score=0.0;
    int num=0;
    std::vector<linemod::Match>::iterator it_match=match_cluster.begin();
    for(;it_match!=match_cluster.end();++it_match)
    {
        sum_score+=it_match->similarity;
        num++;
    }
    sum_score/=num;
    return sum_score;
}

// compute depth and normal diff for 1 cluster
double rgbdDetector::depth_normal_diff_calc(RendererIterator *renderer_iterator_,Matx33d& K_rgb,vector<Mat>& Rs_,vector<Mat>& Ts_,std::vector<linemod::Match> match_cluster, Mat& depth_img)
{
    double sum_depth_diff=0.0;
    double sum_normal_diff=0.0;
    std::vector<linemod::Match>::iterator it_match=match_cluster.begin();
    for(;it_match!=match_cluster.end();++it_match)
     {
        //get mask during rendering
        //get the pose
        cv::Matx33d R_match = Rs_[it_match->template_id].clone();// rotation of the object w.r.t to camera
        cv::Vec3d T_match = Ts_[it_match->template_id].clone();//negative position of the camera with respect to object

        //get the point cloud of the rendered object model
        cv::Mat template_mask,flip_template_mask;
        cv::Rect mask_rect,flip_mask_rect;
        cv::Matx33d R_temp(R_match.inv());//rotation of the viewpoint w.r.t the object
        cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));//up vector (negative y axis)
        cv::Mat depth_template,flip_depth_template;
        renderer_iterator_->renderDepthOnly(depth_template,template_mask, mask_rect, -T_match, up);//
        mask_rect.x = it_match->x;
        mask_rect.y = it_match->y;
        cv::flip(template_mask,flip_template_mask,0);
        cv::flip(depth_template,flip_depth_template,0);
//        cv::imshow("flip template mask",flip_template_mask);
//        cv::waitKey(0);

        //Template mask rect
        //Save rect
        int min_x=0;
        int min_y=0;

        //Search min y
        for(int i=0;i<flip_template_mask.rows;++i)
        {
          bool found_min_y=false;
          for(int j=0;j<flip_template_mask.cols;++j)
          {
            if(flip_template_mask.at<uchar>(i,j) == 255)
              {
                min_y=i;
                found_min_y=true;
                break;
              }
          }
          if(found_min_y)
            break;
        }

        //Search min x
        for(int j =0;j<flip_template_mask.cols;++j)
        {
          bool found_min_x=false;
          for(int i =0;i<flip_template_mask.rows;++i)
          {
            if(flip_template_mask.at<uchar>(i,j) == 255)
            {
              min_x=j;
              found_min_x=true;
              break;
            }

          }
          if(found_min_x)
          {
            break;
          }
        }
        flip_mask_rect=mask_rect;
        flip_mask_rect.y = min_y;
        flip_mask_rect.x = min_x;
        Mat flip_template_mask_crop = flip_template_mask(flip_mask_rect);
        Mat flip_depth_template_crop = flip_depth_template(flip_mask_rect);
        //Compute depth diff for each match
        flip_mask_rect.y = it_match->y;
        flip_mask_rect.x = it_match->x;
        sum_depth_diff+=depth_diff(depth_img,flip_depth_template_crop,flip_template_mask_crop,flip_mask_rect);

        //Compute normal diff for each match
        cv::Rect flip_template_mask_rect = flip_mask_rect;
        flip_template_mask_rect.y = min_y;
        flip_template_mask_rect.x = min_x;
        sum_normal_diff+=normal_diff(depth_img,flip_depth_template,flip_template_mask_crop,flip_template_mask_rect,flip_mask_rect,K_rgb);
    }
    sum_depth_diff=sum_depth_diff/match_cluster.size();
    sum_normal_diff=sum_normal_diff/match_cluster.size();
    int p=0;
    return (getClusterScore(sum_depth_diff,sum_normal_diff));
}

double rgbdDetector::depth_diff(Mat& depth_img,Mat& depth_template,cv::Mat& template_mask,cv::Rect& rect)
{
    //Crop ROI in depth image according to the rect
    Mat depth_roi=depth_img(rect);
    //Convert ROI into a mask. NaN point of depth image will become zero in mask image
    Mat depth_mask,template_mask2;
    depth_roi.convertTo(depth_mask,CV_8UC1,1,0);
    depth_template.convertTo(template_mask2,CV_8UC1);
    //And operation. Only valid points in both images will be considered.
    Mat mask;
    bitwise_and(template_mask,depth_mask,mask);
    cv::imshow("template mask",template_mask);
    cv::imshow("template mask2",template_mask2);
    cv::imshow("depth_mask",depth_mask);
//    cv::imshow("bitwise_mask",mask);
//    cv::waitKey(0);

    //Perform subtraction and accumulate differences
    //-------- Method A for computing depth diff----//
    Mat subtraction(depth_roi.size(),CV_16SC1);
    subtraction=depth_template-depth_roi;
    MatIterator_<uchar> it_mask=mask.begin<uchar>();
    MatIterator_<short> it_subs=subtraction.begin<short>();
    double sum=0.0;
    int num=0;
    for(;it_mask!=mask.end<uchar>();++it_mask,++it_subs)
    {
        if(*it_mask>0)
        {
            //sum=sum+(double)(1/(abs(*it_subs)+1));
            sum=sum+(double)(abs(*it_subs));
            num++;
        }

    }
    sum=(sum/(num*1000.0));
    //-------Method B for computing depth diff----//
//                 t1=cv::getTickCount();
//                 Mat subtraction=Mat::zeros(depth_template.size(),CV_16SC1);
//                 subtract(depth_template,depth_roi,subtraction,mask);
//                 Mat abs_sub=cv::abs(subtraction);
//                 Scalar sumOfDiff=sum(abs_sub);
//                 sum_depth_diff+=sumOfDiff[0];
//                 t1=(cv::getTickCount()-t1)/cv::getTickFrequency();
//                 cout<<"Time consumed by compute depth diff for 1 template: "<<t1<<endl;
    return sum;
}

double rgbdDetector::normal_diff(cv::Mat& depth_img,Mat& depth_template,cv::Mat& template_mask,cv::Rect& template_mask_rect, cv::Rect& rect,Matx33d& K_rgb)
{
    //Convert ROI into a mask. NaN point of depth image will become zero in mask image
    Mat depth_roi=depth_img(rect);
    Mat depth_mask;
    depth_roi.convertTo(depth_mask,CV_8UC1,1,0);
//    depth_mask=Mat::zeros(depth_img.rows,depth_img.cols,CV_8UC1);
//    depth_mask(rect)=255;
    //And operation. Only valid points in both images will be considered.
    Mat mask;
    bitwise_and(template_mask,depth_mask,mask);

    //Type conversion
    Mat depth_roi_64f;
    depth_roi.convertTo(depth_roi_64f,CV_64F);
    Mat depth_template_64f;
    depth_template.convertTo(depth_template_64f,CV_64F);

    //Normal estimation for template depth image
   cv::RgbdNormals template_normal_est(depth_template.rows,depth_template.cols,CV_32F,K_rgb,7,cv::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
   Mat template_normal,roi_template_normal;
   template_normal_est(depth_template,template_normal);
   roi_template_normal = template_normal(template_mask_rect);

   //Normal Estimation for roi depth image
   cv::RgbdNormals roi_normal_est(depth_img.rows,depth_img.cols,CV_32F,K_rgb,7,cv::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
   Mat depth_img_normal,roi_normal;
   roi_normal_est(depth_img,depth_img_normal);
   roi_normal=depth_img_normal(rect);


   //Compute normal diff
   Mat subtraction=roi_normal-roi_template_normal;
   MatIterator_<uchar> it_mask=mask.begin<uchar>();
   MatIterator_<Vec3d> it_subs=subtraction.begin<Vec3d>();
   double sum=0.0;
   int num=0;
//   for(;it_mask!=mask.end<uchar>();++it_mask,++it_subs)
//   {
//       if(*it_mask>0)
//       {
//           if(isValidDepth((*it_subs)[0]) && isValidDepth((*it_subs)[1]) && isValidDepth((*it_subs)[2]))
//           {
//               double abs_diff=abs((*it_subs)[0])+abs((*it_subs)[1])+abs((*it_subs)[2]);
//               //sum+=1/(abs_diff+1);
//               sum+=abs_diff;
//               num++;
//           }


//       }

//   }
   for(int i=0;i<mask.rows;i++)
   {
       for(int j=0;j<mask.cols;j++)
       {
           if(mask.at<uchar>(i,j)>0)
           {
               cv::Vec3f template_normal_vec_cv = roi_template_normal.at<Vec3f>(i,j);
               cv::Vec3f roi_normal_vec_cv = roi_normal.at<Vec3f>(i,j);
               if(abs(cv::norm(template_normal_vec_cv)-1.0)<0.0001 && abs(cv::norm(roi_normal_vec_cv)-1.0)<0.0001)
               {
                   Eigen::Vector4f model_normal_vec((float)roi_template_normal.at<Vec3f>(i,j)[0],(float)roi_template_normal.at<Vec3f>(i,j)[1],(float)roi_template_normal.at<Vec3f>(i,j)[2],0.0);
                   Eigen::Vector4f roi_normal_vec((float)roi_normal.at<Vec3f>(i,j)[0],(float)roi_normal.at<Vec3f>(i,j)[1],(float)roi_normal.at<Vec3f>(i,j)[2],0.0);
                   sum+=pcl::getAngle3D(model_normal_vec,roi_normal_vec);
                   num++;
               }
           }
       }
   }

   sum/=num;
   return sum;

}

void rgbdDetector::nonMaximaSuppression(vector<ClusterData>& cluster_data,const double& neighborSize, vector<Rect>& Rects_,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match)
{
    vector<ClusterData> nms_cluster_data;
    std::map<std::vector<int>, std::vector<linemod::Match> > nms_map_match;
    vector<ClusterData>::iterator it1=cluster_data.begin();
    for(;it1!=cluster_data.end();++it1)
    {
        if(!it1->is_checked)
        {            
            ClusterData* best_cluster=&(*it1);
            vector<ClusterData>::iterator it2=it1;
            it2++;
            //Look for local maxima
            for(;it2!=cluster_data.end();++it2)
            {
                if(!it2->is_checked)
                {
                    double dist=sqrt((best_cluster->index[0]-it2->index[0]) * (best_cluster->index[0]-it2->index[0]) + (best_cluster->index[1]-it2->index[1]) * (best_cluster->index[1]-it2->index[1]));
                    if(dist < neighborSize)
                    {
                        it2->is_checked=true;
                        if(it2->score > best_cluster->score)
                        {
                            best_cluster=&(*it2);
                        }
                    }
                }
            }
            nms_cluster_data.push_back(*best_cluster);
        }
    }

    cluster_data.clear();
    cluster_data=nms_cluster_data;

    //Add matches to cluster data
    it1=cluster_data.begin();
    for(;it1!=cluster_data.end();++it1)
    {
        std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it2;
        it2=map_match.find(it1->index);
        nms_map_match.insert(*it2);
        it1->matches=it2->second;
    }

    //Compute bounding box for each cluster
    it1=cluster_data.begin();
//    for(;it1!=cluster_data.end();++it1)
//    {
//        int X=0; int Y=0; int WIDTH=0; int HEIGHT=0;
//        std::vector<linemod::Match>::iterator it2=it1->matches.begin();
//        for(;it2!=it1->matches.end();++it2)
//        {
//            Rect tmp=Rects_[it2->template_id];
//            X+=it2->x;
//            Y+=it2->y;
//            WIDTH+=tmp.width;
//            HEIGHT+=tmp.height;
//        }
//        X/=it1->matches.size();
//        Y/=it1->matches.size();
//        WIDTH/=it1->matches.size();
//        HEIGHT/=it1->matches.size();

//        it1->rect=Rect(X,Y,WIDTH,HEIGHT);

//    }

    for(;it1!=cluster_data.end();++it1)
    {
        int X=0; int Y=0; int WIDTH=0; int HEIGHT=0; int max_width=0; int max_height=0;
        std::vector<linemod::Match>::iterator it2=it1->matches.begin();
        for(;it2!=it1->matches.end();++it2)
        {
            Rect tmp=Rects_[it2->template_id];
            X+=it2->x;
            Y+=it2->y;
            //Use max width/height
//            if(tmp.width>max_width)
//                max_width=tmp.width;
//            if(tmp.height>max_height)
//                max_height=tmp.height;

            //Use average width/height
            WIDTH+=tmp.width;
            HEIGHT+=tmp.height;
        }
        X/=it1->matches.size();
        Y/=it1->matches.size();
        WIDTH/=it1->matches.size();
        HEIGHT/=it1->matches.size();

        it1->rect=Rect(X,Y,WIDTH,HEIGHT);

    }

    map_match.clear();
    map_match=nms_map_match;
    int p=0;
}

void rgbdDetector::nonMaximaSuppressionUsingIOU(vector<ClusterData>& cluster_data,const double& neighborSize, vector<Rect>& Rects_,std::map<std::vector<int>, std::vector<linemod::Match> >& map_match)
{

  //Add matches to cluster data
  vector<ClusterData>::iterator it1=cluster_data.begin();
  for(;it1!=cluster_data.end();++it1)
  {
      std::map<std::vector<int>, std::vector<linemod::Match> >::iterator it2;
      it2=map_match.find(it1->index);
      it1->matches=it2->second;

      //Compute BBox
      int X=0; int Y=0; int WIDTH=0; int HEIGHT=0; int max_width=0; int max_height=0;
      std::vector<linemod::Match>::iterator it3=it1->matches.begin();
      for(;it3!=it1->matches.end();++it3)
      {
          Rect tmp=Rects_[it3->template_id];
          X+=it3->x;
          Y+=it3->y;
          //Use average width/height
          WIDTH+=tmp.width;
          HEIGHT+=tmp.height;
      }
      X/=it1->matches.size();
      Y/=it1->matches.size();
      WIDTH/=it1->matches.size();
      HEIGHT/=it1->matches.size();
      it1->rect=Rect(X,Y,WIDTH,HEIGHT);

  }

  //Sort scores
  std::sort(cluster_data.begin(),cluster_data.end(),sortScoreCluster);

  //Non-maxima suppression
  it1=cluster_data.begin();
  std::map<std::vector<int>, std::vector<linemod::Match> > nms_map_match;
  for(;it1!=cluster_data.end();++it1)
  {
      if(!it1->is_checked)
      {
          vector<ClusterData>::iterator it2=it1;
          it2++;
          //Look for local maxima
          for(;it2!=cluster_data.end();++it2)
          {
              if(!it2->is_checked)
              {
                  double IoU=computeIoU(it1->rect, it2->rect);
                  if(IoU > 0.4)
                  {
                      it2->is_checked=true;
                  }
              }
          }
      }
  }

  vector<ClusterData> nms_cluster_data;
  it1=cluster_data.begin();
  for(;it1!=cluster_data.end();++it1)
  {
    if(!it1->is_checked)
      nms_cluster_data.push_back(*it1);
  }

  cluster_data.clear();
  cluster_data=nms_cluster_data;
}

float rgbdDetector::computeIoU(cv::Rect rect1, cv::Rect rect2)
{
  int rect1_minX, rect1_minY, rect1_maxX, rect1_maxY;
  int rect2_minX, rect2_minY, rect2_maxX, rect2_maxY;

  rect1_minX = rect1.x;
  rect1_maxX = rect1.x + rect1.width-1;
  rect1_minY = rect1.y;
  rect1_maxY = rect1.y + rect1.height-1;

  rect2_minX = rect2.x;
  rect2_maxX = rect2.x + rect2.width-1;
  rect2_minY = rect2.y;
  rect2_maxY = rect2.y + rect2.height-1;

  int minX = MAX(rect1_minX, rect2_minX);
  int maxX = MIN(rect1_maxX, rect2_maxX);
  int minY = MAX(rect1_minY, rect2_minY);
  int maxY = MIN(rect1_maxY, rect2_maxY);

  bool is_x_inter = false;
  bool is_y_inter = false;

  if( (minX >= rect1_minX && minX <= rect1_maxX) || (minX >= rect2_minX && minX <= rect2_maxX) )
    is_x_inter = true;
  if( (minY >= rect1_minY && minY <= rect1_maxY) || (minY >= rect2_minY && minY <= rect2_maxY) )
    is_y_inter = true;

  float inter_area;
  if(is_x_inter && is_y_inter)
  {
    inter_area = (maxX - minX + 1) * (maxY - minY +1);
  }
  else
  {
    inter_area = 0.0;
  }
  float union_area = rect1.width * rect1.height + rect2.width * rect2.height - inter_area;

  float IoU = inter_area / union_area;

  return IoU;
}

double rgbdDetector::getClusterScore(const double& depth_diff_score,const double& normal_diff_score)
{
    //Simply add two scores
    //return(depth_diff_score+normal_diff_score);
    double depth_similarity = 1/exp(depth_diff_score);
    double normal_similarity=1/exp(normal_diff_score);
    return(depth_similarity*normal_similarity);

}

void rgbdDetector::getRoughPoseByClustering(vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc,vector<Mat>& Rs_,vector<Mat>& Ts_, vector<double>& Distances_,vector<double>& Obj_origin_dists,float orientation_clustering_th_,RendererIterator *renderer_iterator_,double& renderer_focal_length_x,double& renderer_focal_length_y,IMAGE_WIDTH& image_width,int& bias_x)
{
    //For each cluster
    for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
    {
        //Perform clustering
        vector<vector<Eigen::Matrix3d> > orienClusters;
        vector<vector<int> > idClusters;
        vector<vector<pair<int ,int> > > xyClusters;
        for(vector<linemod::Match>::iterator it2=it->matches.begin();it2!=it->matches.end();++it2)
        {
            //Get rotation
            Mat R_mat=Rs_[it2->template_id];
            Eigen::Matrix<double,3,3> R;
            cv2eigen(R_mat,R);

            bool found_cluster=false;

            //Compare current orientation with existing cluster
            for(int i=0;i<orienClusters.size();++i)
            {
                if(orientationCompare(R,orienClusters[i].front(),orientation_clustering_th_))
                {
                    found_cluster=true;
                    orienClusters[i].push_back(R);
                    idClusters[i].push_back(it2->template_id);
                    xyClusters[i].push_back(pair<int ,int>(it2->x,it2->y));
                    break;
                }
            }

            //If current orientation is not assigned to any cluster, create a new one for it.
            if(found_cluster == false)
            {
                vector<Eigen::Matrix3d> new_cluster;
                new_cluster.push_back(R);
                orienClusters.push_back(new_cluster);

                vector<int> new_cluster_;
                new_cluster_.push_back(it2->template_id);
                idClusters.push_back(new_cluster_);

                vector<pair<int,int> > new_xy_cluster;
                new_xy_cluster.push_back(pair<int,int>(it2->x,it2->y));
                xyClusters.push_back(new_xy_cluster);
            }
        }
        //Sort cluster according to the number of poses
        std::sort(orienClusters.begin(),orienClusters.end(),sortOrienCluster);
        std::sort(idClusters.begin(),idClusters.end(),sortIdCluster);
        std::sort(xyClusters.begin(),xyClusters.end(),sortXyCluster);

        //Test display all the poses in 1st cluster
        for(int i=0;i<idClusters[0].size();++i)
        {
            int ID=idClusters[0][i];
            //get the pose
            cv::Matx33d R_match = Rs_[ID].clone();// rotation of the object w.r.t to the view point
            cv::Vec3d T_match = Ts_[ID].clone();//the translation of the camera with respect to the current view point
            //cv::Mat K_matrix= Ks_[ID].clone();
            cv::Mat mask;
            cv::Rect rect;
            cv::Matx33d R_temp(R_match.inv());//rotation of the viewpoint w.r.t the object
            cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));//the negative direction of y axis of viewpoint w.r.t object frame
            cv::Mat depth_ref_, rgb_ref_;
            renderer_iterator_->renderDepthOnly(depth_ref_, mask, rect, -T_match, up);
            renderer_iterator_->renderImageOnly(rgb_ref_, rect, -T_match, up);
//            imshow("mask",mask);
//            imshow("rgb image",rgb_ref_);
//            waitKey(0);
        }

       //Average all poses in 1st cluster
        Eigen::Vector3d T_aver(0.0,0.0,0.0);
        Eigen::Vector4d R_aver(0.0,0.0,0.0,0.0);
        double D_aver=0.0;
        double Trans_aver=0.0;
        vector<Eigen::Matrix3d>::iterator iter=orienClusters[0].begin();
        bool is_center_hole=false;
        int X=0; int Y=0;
        for(int i=0;iter!=orienClusters[0].end();++iter,++i)
        {
            //get rotation
               //Matrix33d ---> Quaternion ---> Vector4d
            R_aver+=Eigen::Quaterniond(*iter).coeffs();
            //get translation
            Mat T_mat=Ts_[idClusters[0][i]];
            Eigen::Matrix<double,3,1> T;
            cv2eigen(T_mat,T);
            T_aver+=T;
            //Get distance
            D_aver+=Distances_[idClusters[0][i]];
            //Get translation
            Trans_aver+=Obj_origin_dists[idClusters[0][i]];
            //Get match position
            X+=xyClusters[0][i].first;
            Y+=xyClusters[0][i].second;

            if(fabs(Distances_[idClusters[0][i]]-Obj_origin_dists[idClusters[0][i]])<0.001)
            {
               is_center_hole=true;
            }
        }

       //Averaging operation
        R_aver/=orienClusters[0].size();
        T_aver/=orienClusters[0].size();
        D_aver/=orienClusters[0].size();
        Trans_aver/=orienClusters[0].size();
        X/=orienClusters[0].size();
        Y/=orienClusters[0].size();
       //normalize the averaged quaternion
        Eigen::Quaterniond quat=Eigen::Quaterniond(R_aver).normalized();
        cv::Mat R_mat;
        Eigen::Matrix3d R_eig;
        R_eig=quat.toRotationMatrix();
        eigen2cv(R_eig,R_mat);

//        if(fabs(D_aver-Trans_aver) < 0.001)
//            {
//            is_center_hole=true;
//        }

        cv::Mat mask,flip_mask;
        cv::Rect mask_rect,flip_mask_rect;
        cv::Matx33d R_match(R_mat); //object orientaton
        cv::Matx33d R_cam = R_match.inv();
        cv::Vec3d up(-R_cam(0,1), -R_cam(1,1), -R_cam(2,1));//the negative direction of y axis of viewpoint w.r.t object frame
        cv::Mat depth_render, flip_depth_render;
        cv::Mat image_render;
        cv::Vec3d T_match;
        T_match[0]=T_aver[0];
        T_match[1]=T_aver[1];
        T_match[2]=T_aver[2];
        renderer_iterator_->renderDepthOnly(depth_render, mask, mask_rect, -T_match, up);
        renderer_iterator_->renderImageOnly(image_render, mask_rect, -T_match, up);
        cv::flip(depth_render,flip_depth_render,0);
        cv::flip(mask,flip_mask,0);
//        imshow("flip_mask oupt",flip_mask);
//       imshow("mask output",mask);
//       imshow("rgb image output", image_render);
//       waitKey(0);

        //Save mask
        it->mask=flip_mask;

        //Save rect
        int min_x=0;
        int min_y=0;

        //Search min y
        for(int i=0;i<flip_mask.rows;++i)
        {
          bool found_min_y=false;
          for(int j=0;j<flip_mask.cols;++j)
          {
            if(flip_mask.at<uchar>(i,j) == 255)
              {
                min_y=i;
                found_min_y=true;
                break;
              }
          }
          if(found_min_y)
            break;
        }

        //Search min x
        for(int j =0;j<flip_mask.cols;++j)
        {
          bool found_min_x=false;
          for(int i =0;i<flip_mask.rows;++i)
          {
            if(flip_mask.at<uchar>(i,j) == 255)
            {
              min_x=j;
              found_min_x=true;
              break;
            }

          }
          if(found_min_x)
          {
            break;
          }
        }


        flip_mask_rect=mask_rect;
        flip_mask_rect.y = min_y;
        flip_mask_rect.x = min_x;
        it->rect.width=mask_rect.width;
        it->rect.height=mask_rect.height;
        it->rect.x=X;
        it->rect.y=Y;

        //Save orientation, T_match and K_matrix
        it->orientation=R_match;
        it->T_match=T_match;
        it->dist=D_aver;
        it->K_matrix=(Mat_<double>(3,3)<<renderer_focal_length_x,0.0,depth_render.cols/2,
                                      0.0,renderer_focal_length_y,depth_render.rows/2,
                                      0.0,0.0,1.0);

        it->pose.linear()=R_eig;
        it->pose.translation()<<0.0,0.0,Trans_aver;


        //Get render point cloud
        Mat pc_cv;
//        cv::depthTo3d(flip_depth_render,it->K_matrix,pc_cv);    //mm ---> m
        cv::depthTo3d(flip_depth_render, it->K_matrix, pc_cv ,noArray());
        it->model_pc->header.frame_id="/camera_link";
        for(int ii=0;ii<pc_cv.rows;++ii)
        {
            double* row_ptr=pc_cv.ptr<double>(ii);
            for(int jj=0;jj<pc_cv.cols;++jj)
            {
                if(flip_mask.at<uchar>(ii,jj)>0)
                {
                    double* data =row_ptr+jj*3;
                    it->model_pc->points.push_back(pcl::PointXYZ(data[0],data[1],data[2]));
                }
            }
        }
        it->model_pc->height=1;
        it->model_pc->width=it->model_pc->points.size();

        //Get scene point cloud indices
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        indices=getPointCloudIndices(it,image_width,bias_x,flip_mask_rect);
        //Extract scene pc according to indices
        PointCloudXYZ::Ptr scene_pc(new PointCloudXYZ);
        extractPointsByIndices(indices,pc,scene_pc,false,false);

        //Viz for test
//        pcl::visualization::PCLVisualizer view("v");
//        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_pc_color(it->model_pc,0,255,0);
//        view.addPointCloud(it->model_pc,model_pc_color,"model");
//        view.addPointCloud(scene_pc,"scene");
//        Eigen::Affine3f obj_pose_f=it->pose.cast<float>();
//        view.addCoordinateSystem(0.08,obj_pose_f);
//        view.spin();

        //Remove Nan points
        vector<int> index;
        it->model_pc->is_dense=false;
        pcl::removeNaNFromPointCloud(*(it->model_pc),*(it->model_pc),index);
        pcl::removeNaNFromPointCloud(*scene_pc,*scene_pc,index);
        //Statistical outlier removal
        statisticalOutlierRemoval(scene_pc,50,1.0);

        //Other preprocessing technique
//        euclidianClustering(scene_pc,0.01);
        //Save dense scene pc first
        pcl::copyPointCloud(*scene_pc,*(it->dense_scene_pc));
        float leaf_size=0.002;
        voxelGridFilter(scene_pc,leaf_size);
        voxelGridFilter(it->model_pc,leaf_size);

        //Save scene point cloud for later HV
        it->scene_pc=scene_pc;

        //Get position
        Eigen::Vector3d position(0.0,0.0,0.0);
        Eigen::Affine3d transform;

        //getPositionByROICenter(it,pc,x,y,Trans_aver,transform,position);
        //getPositionByDistanceOffset (it,pc,x,y,image_width,bias_x,D_aver,Trans_aver,is_center_hole,position,transform);
        getPositionBySurfaceCentroid (it->scene_pc,it->model_pc,Trans_aver,position,transform);
        //getPoseByLocalDescriptor (it->scene_pc,it->model_pc,Trans_aver,transform,position);

        //Tranlsate pointcloud
        it->pose.translation()<< position[0],position[1],position[2];
        //it->pose.linear() = transform.linear () * it->pose.linear();
        PointCloudXYZ::Ptr transformed_cloud (new PointCloudXYZ ());
        pcl::transformPointCloud(*it->model_pc,*transformed_cloud,transform);
        pcl::copyPointCloud(*transformed_cloud,*it->model_pc);
    }
}

void rgbdDetector::getPoseByLocalDescriptor(PointCloudXYZ::Ptr scene_pc,PointCloudXYZ::Ptr model_pc,double DistanceFromCamToObj,Eigen::Affine3d& transform,Eigen::Vector3d& position)
{
    //Params
    int k_neigh_norest = 50;
    float radius_unisamp = 0.002f;
    float descr_rad_ = 0.02f;
    float rf_rad_ = 0.015f;
    float cg_size_ = 0.01f;
    float cg_thresh_ = 5.0f;

    //Normal estimation
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
    norm_est.setKSearch (k_neigh_norest);
        //scene
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
    norm_est.setInputCloud (scene_pc);
    norm_est.compute (*scene_normals);
        //model
    pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
    norm_est.setInputCloud (model_pc);
    norm_est.compute (*model_normals);

    //Keypoints detection: Uniform sampling
        //scene
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud (*scene_pc,*scene_keypoints);
    voxelGridFilter (scene_keypoints,radius_unisamp);
        //model
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud (*model_pc,*model_keypoints);
    voxelGridFilter (model_keypoints,radius_unisamp);

    //SHOT description
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descr_est;
    descr_est.setRadiusSearch (descr_rad_);
        //model
    pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    descr_est.setInputCloud (model_keypoints);
    descr_est.setInputNormals (model_normals);
    descr_est.setSearchSurface (model_pc);
    descr_est.compute (*model_descriptors);
        //scene
        pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
    descr_est.setInputCloud (scene_keypoints);
    descr_est.setInputNormals (scene_normals);
    descr_est.setSearchSurface (scene_pc);
    descr_est.compute (*scene_descriptors);

    //  Find Model-Scene Correspondences with KdTree
    //
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    pcl::KdTreeFLANN<pcl::SHOT352> match_search;
    match_search.setInputCloud (model_descriptors);

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < scene_descriptors->size (); ++i)
    {
      std::vector<int> neigh_indices (1);
      std::vector<float> neigh_sqr_dists (1);
      if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
      {
        continue;
      }
      int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
      if(found_neighs == 1 && neigh_sqr_dists[0] < 0.5f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
      {
        pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
        model_scene_corrs->push_back (corr);
      }
    }

    //  Compute (Keypoints) Reference Frames only for Hough
    //
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf (new pcl::PointCloud<pcl::ReferenceFrame> ());
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf (new pcl::PointCloud<pcl::ReferenceFrame> ());

    pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::Normal, pcl::ReferenceFrame> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model_pc);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene_pc);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);

    //Compute position
    transform.matrix ()=rototranslations[0].cast<double>();
    position = Eigen::Vector3d(0.0+transform(0,3),0.0+transform(1,3),DistanceFromCamToObj+transform(2,3));
}

void rgbdDetector::getPositionByDistanceOffset(vector<ClusterData>::iterator it,PointCloudXYZ::Ptr pc,int x_index,int y_index,IMAGE_WIDTH image_width,int bias_x,double DistanceOffset,double DistanceFromCamToObj,bool is_center_hole,Eigen::Vector3d& position, Eigen::Affine3d& transform,cv::Rect mask_rect)
{
    pcl::PointXYZ bbox_center =pc->at(x_index,y_index);
    //Deal with the situation that there is a hole in ROI or in the model pointcloud during rendering
    if(pcl_isnan(bbox_center.x) || pcl_isnan(bbox_center.y) || pcl_isnan(bbox_center.z) || is_center_hole)
    {
        PointCloudXYZ::Ptr pts_tmp(new PointCloudXYZ());
        pcl::PointIndices::Ptr indices_tmp(new pcl::PointIndices());
        vector<int> index_tmp;

        indices_tmp=getPointCloudIndices(it,image_width,bias_x,mask_rect);
        extractPointsByIndices(indices_tmp,pc,pts_tmp,false,false);
        pcl::removeNaNFromPointCloud(*pts_tmp,*pts_tmp,index_tmp);

        //Comptute centroid
        Eigen::Matrix<double,4,1> centroid;
        if(pcl::compute3DCentroid<pcl::PointXYZ,double>(*pts_tmp,centroid)!=0)
        {
            //Replace the Nan center point with centroid
            bbox_center.x=centroid[0];
            bbox_center.y=centroid[1];
            bbox_center.z=centroid[2];
        }
        else
        {
           ROS_ERROR("Pose clustering: unable to compute centroid of the pointcloud!");
        }
    }

    //Notice: No hole in the center
    if(!is_center_hole)
    {
        bbox_center.z+=DistanceOffset;
    }

    position = Eigen::Vector3d(bbox_center.x,bbox_center.y,bbox_center.z);

    //Compute translation
    pcl::PointXYZ scene_pt(bbox_center.x,bbox_center.y,bbox_center.z);
    pcl::PointXYZ model_pt(0,0,DistanceFromCamToObj);
    pcl::PointXYZ translation(scene_pt.x -model_pt.x,scene_pt.y -model_pt.y,scene_pt.z -model_pt.z);
    transform =Eigen::Affine3d::Identity();
    transform.translation()<<translation.x, translation.y, translation.z;
}

void rgbdDetector::getPositionByROICenter(vector<ClusterData>::iterator it,PointCloudXYZ::Ptr pc,int x_index_scene,int y_index_scene,double DistanceFromCamToObj,Eigen::Affine3d& transform,Eigen::Vector3d& position)
{
    pcl::PointXYZ scene_pt =pc->at(x_index_scene,y_index_scene);
    int x_index_model=it->rect.width/2;
    int y_index_model=it->rect.height/2;
    pcl::PointXYZ model_pt = it->model_pc->at(x_index_model,y_index_model);
    int offset=0;

    //Only when both scene point and model point are vaild, can the computation of translation continues
    while(pcl_isnan(scene_pt.x) || pcl_isnan(model_pt.x))
    {
        offset++;
        if(offset>it->rect.width/2)
        {
            cout<<"Error: getPositionByCenterSurface"<<endl;
            cout<<"None of the scene-model point pairs is vaild";
            break;
        }
        x_index_scene+=offset;
        x_index_model+=offset;
        scene_pt =pc->at(x_index_scene,y_index_scene);
        model_pt = it->model_pc->at(x_index_model,y_index_model);
    }

    //Compute translation
    pcl::PointXYZ translation(scene_pt.x -model_pt.x,scene_pt.y -model_pt.y,scene_pt.z -model_pt.z);
    transform =Eigen::Affine3d::Identity();
    transform.translation()<<translation.x, translation.y, translation.z;

    //Compute position
    position = Eigen::Vector3d(0.0+translation.x,0.0+translation.y,DistanceFromCamToObj+translation.z);


}

void rgbdDetector::getPositionBySurfaceCentroid(PointCloudXYZ::Ptr scene_pc,PointCloudXYZ::Ptr model_pc,double DistanceFromCamToObj,Eigen::Vector3d& position,Eigen::Affine3d& transform)
{
    //Get scene points centroid
    Eigen::Matrix<double,4,1> scene_centroid;
    pcl::PointXYZ scene_surface_centroid;
    bool is_point_valid=true;
    if(pcl::compute3DCentroid<pcl::PointXYZ,double>(*scene_pc,scene_centroid)!=0)
    {
        //Search nearest point to centroid point
        int K = 1;
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
//        float octree_res=0.001;
//        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(octree_res));
        pcl::KdTreeFLANN<pcl::PointXYZ> tree;

//        octree->setInputCloud(scene_pc);
//        octree->addPointsFromInputCloud();

        tree.setInputCloud(scene_pc);
        if (tree.nearestKSearch (pcl::PointXYZ(scene_centroid[0],scene_centroid[1],scene_centroid[2]), K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            scene_surface_centroid=scene_pc->points[pointIdxNKNSearch[0]];
        }
    }
    else
    {
        cout<<"Error: getPositionBySurfaceCentroid"<<endl;
        cout<<"Can not compute centroid of scene pointcloud!"<<endl;
        is_point_valid=false;
    }

    //Get model points centroid
    Eigen::Matrix<double,4,1> model_centroid;
    pcl::PointXYZ model_surface_centroid;
    if(pcl::compute3DCentroid<pcl::PointXYZ,double>(*model_pc,model_centroid)!=0)
    {
        //Search nearest point to centroid point
        int K = 1;
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
        float octree_res=0.001;
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::Ptr octree(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(octree_res));

        //First, from model points
        octree->setInputCloud(model_pc);
        octree->addPointsFromInputCloud();
        if (octree->nearestKSearch (pcl::PointXYZ(model_centroid[0],model_centroid[1],model_centroid[2]), K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            model_surface_centroid=model_pc->points[pointIdxNKNSearch[0]];
        }
    }
    else
    {
        cout<<"Error: getPositionBySurfaceCentroid"<<endl;
        cout<<"Can not compute centroid of model pointcloud!"<<endl;
        is_point_valid=false;
    }

    if(is_point_valid)
    {
        //Compute translation
        pcl::PointXYZ translation(scene_surface_centroid.x -model_surface_centroid.x,scene_surface_centroid.y -model_surface_centroid.y,scene_surface_centroid.z -model_surface_centroid.z);
        transform =Eigen::Affine3d::Identity();
        transform.translation()<<translation.x, translation.y, translation.z;

        //Compute position
        position = Eigen::Vector3d(0.0+translation.x,0.0+translation.y,DistanceFromCamToObj+translation.z);
    }
}

void rgbdDetector::graspingPoseBasedOnRegionGrowing(PointCloudXYZ::Ptr dense_scene_pc,PointCloudXYZ::Ptr scene_pc,float normal_thresh, float curvature_thresh,double offset,Eigen::Affine3d& grasping_pose,bool is_viz)
{
    //Params
    float k_neigh_norest=50;
    float k_neigh_reg=30;
    grasping_pose=Eigen::Affine3d::Identity ();

    //Denoise
    //statisticalOutlierRemoval(scene_pc,50,1.0);
//    voxelGridFilter(scene_pc,0.002);

    //Smooth
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_(new pcl::search::KdTree<pcl::PointXYZ>);
    PointCloudXYZ::Ptr tmp_pc(new PointCloudXYZ);
    pcl::MovingLeastSquares<pcl::PointXYZ,pcl::PointXYZ> mls;
    mls.setInputCloud(scene_pc);
    mls.setComputeNormals(true);
    mls.setPolynomialFit(true);
    mls.setSearchMethod(tree_);
    mls.setSearchRadius(0.04);
    mls.process(*tmp_pc);
    pcl::copyPointCloud(*tmp_pc,*scene_pc);

    //Normal estimation
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
    norm_est.setKSearch (k_neigh_norest);
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
    norm_est.setInputCloud (scene_pc);
    norm_est.setSearchSurface(dense_scene_pc);
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
    reg.setSmoothnessThreshold (normal_thresh / 180.0 * M_PI);
    reg.setCurvatureThreshold (curvature_thresh);
    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);
    std::sort(clusters.begin(),clusters.end(),sortRegionIndx);
    PointCloudXYZ::Ptr segmented_pc(new PointCloudXYZ);
    extractPointsByIndices (boost::make_shared<pcl::PointIndices>(clusters[0]),scene_pc,segmented_pc,false,false);

    //Viz for test
    if(is_viz)
    {
        pcl::visualization::PCLVisualizer::Ptr v(new pcl::visualization::PCLVisualizer);
        v->addPointCloud<pcl::PointXYZ>(segmented_pc);
        v->spin();
    }

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
            //scene_surface_normal=scene_normals->at(clusters[0].indices[pointIdxNKNSearch[0]]);

            norm_est.setKSearch (k_neigh_norest);
            pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
            norm_est.setInputCloud (segmented_pc);
            norm_est.setSearchSurface(scene_pc);
            norm_est.compute (*scene_normals);
            scene_surface_normal=scene_normals->at(pointIdxNKNSearch[0]);
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

bool rgbdDetector::orientationCompare(Eigen::Matrix3d& orien1,Eigen::Matrix3d& orien2,double thresh)
{
    //lazyProduct and eval reference: https://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html
    Eigen::AngleAxisd rotation_diff_mat (orien1.inverse ().lazyProduct (orien2).eval());
    double angle_diff = (double)(fabs(rotation_diff_mat.angle())/M_PI*180.0);
    //thresh unit: degree
    if(angle_diff<thresh)
    {
        return true;
    }
    else
    {
        return false;
    }

}

void rgbdDetector::icpPoseRefine(vector<ClusterData>& cluster_data,pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>& icp,PointCloudXYZ::Ptr pc, IMAGE_WIDTH image_width,int bias_x,bool is_viz)
{
    pcl::visualization::PCLVisualizer::Ptr v;
    if(is_viz)
    {
        v=boost::make_shared<pcl::visualization::PCLVisualizer>("view");
        //v->setBackgroundColor(255,255,255);
    }
    int id=0;
    for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
    {
        //PreProcessing
        //        //Remove Nan points
        //        vector<int> index;
        //        it->model_pc->is_dense=false;
        //        pcl::removeNaNFromPointCloud(*(it->model_pc),*(it->model_pc),index);
        //        pcl::removeNaNFromPointCloud(*scene_pc,*scene_pc,index);
        //Statistical outlier removal
        //statisticalOutlierRemoval(it->scene_pc,50,1.0);
        //Smooth
//        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_(new pcl::search::KdTree<pcl::PointXYZ>);
//        PointCloudXYZ::Ptr tmp_pc(new PointCloudXYZ);
//        pcl::MovingLeastSquares<pcl::PointXYZ,pcl::PointXYZ> mls;
//        mls.setInputCloud(it->scene_pc);
//        mls.setComputeNormals(true);
//        mls.setPolynomialFit(true);
//        mls.setSearchMethod(tree_);
//        mls.setSearchRadius(0.01);
//        mls.process(*tmp_pc);
//        pcl::copyPointCloud(*tmp_pc,*(it->scene_pc));
        //Downsample
//        voxelGridFilter(it->scene_pc,0.002);
//        voxelGridFilter(it->model_pc,0.002);
        //        //euclidianClustering(scene_pc,0.01);

//        //Viz for test
        if(is_viz)
        {
            string model_str="model";
            string scene_str="scene";
            stringstream ss;
            ss<<id;
            model_str+=ss.str();
            scene_str+=ss.str();
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_pc_color(it->model_pc,0,255,0);
            v->addPointCloud(it->scene_pc,scene_str);
            v->addPointCloud(it->model_pc,model_pc_color,model_str);
            v->spin();
        }

        //Coarse alignment
        icp.setRANSACOutlierRejectionThreshold(0.02);
        icp.setInputSource (it->model_pc);
        icp.setInputTarget (it->scene_pc);
        icp.align (*(it->model_pc));
        if(!icp.hasConverged())
        {
            cout<<"ICP cannot converge"<<endl;
        }
        else{
            //cout<<"ICP fitness score of coarse alignment: "<<icp.getFitnessScore()<<endl;
        }
        //Update pose
        Eigen::Matrix4f tf_mat = icp.getFinalTransformation();
        Eigen::Matrix4d tf_mat_d=tf_mat.cast<double>();
        Eigen::Affine3d tf(tf_mat_d);
        it->pose=tf*it->pose;

        //Viz for test
        if(is_viz)
        {
            string model_str="model";
            string scene_str="scene";
            stringstream ss;
            ss<<id;
            model_str+=ss.str();
            scene_str+=ss.str();
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);            
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_pc_color(it->model_pc,0,255,0);
            v->updatePointCloud(it->scene_pc,scene_str);
            v->updatePointCloud(it->model_pc,model_pc_color,model_str);
            v->spin();
        }

        //Fine alignment 1
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACOutlierRejectionThreshold(0.01);
        icp.setMaximumIterations(20);
        icp.setMaxCorrespondenceDistance(0.01);
        icp.setInputSource (it->model_pc);
        icp.setInputTarget (it->scene_pc);
        icp.align (*(it->model_pc));
        if(!icp.hasConverged())
        {
            cout<<"ICP cannot converge"<<endl;
        }
        else{
            //cout<<"ICP fitness score of fine alignment 1: "<<icp.getFitnessScore()<<endl;
        }
        //Update pose
        tf_mat = icp.getFinalTransformation();
        tf_mat_d=tf_mat.cast<double>();
        tf.matrix()=tf_mat_d;
        it->pose=tf*it->pose;

        //Viz for test
        if(is_viz)
        {
            string model_str="model";
            string scene_str="scene";
            stringstream ss;
            ss<<id;
            model_str+=ss.str();
            scene_str+=ss.str();
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_pc_color(it->model_pc,0,255,0);
            v->updatePointCloud(it->scene_pc,scene_str);
            v->updatePointCloud(it->model_pc,model_pc_color,model_str);
            v->spin();
        }

//                 //Fine alignment 2
//                 icp.setRANSACOutlierRejectionThreshold(0.005);
//                 icp.setMaximumIterations(10);
//                 icp.setMaxCorrespondenceDistance(0.005);
//                 icp.setInputSource (it->model_pc);
//                 icp.setInputTarget (scene_pc);
//                 icp.align (*(it->model_pc));
//                 if(!icp.hasConverged())
//                 {
//                     cout<<"ICP cannot converge"<<endl;
//                 }
//                 else{
//                     cout<<"ICP fitness score of fine alignment 2: "<<icp.getFitnessScore()<<endl;
//                 }
//                 //Update pose
//                 tf_mat = icp.getFinalTransformation();
//                 tf_mat_d=tf_mat.cast<double>();
//                 tf.matrix()=tf_mat_d;
//                 it->pose=tf*it->pose;

//                 //Viz test
//                 v.updatePointCloud(it->model_pc,color,"model");
//                 v.spin();

        id++;

    }
}

void rgbdDetector::euclidianClustering(PointCloudXYZ::Ptr pts,float dist)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (pts);
    std::vector<int> index1;
    pcl::removeNaNFromPointCloud(*pts,*pts,index1);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (dist); // 1cm
    ec.setMinClusterSize (50);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (pts);
    ec.extract (cluster_indices);
    PointCloudXYZ::Ptr pts_filtered(new PointCloudXYZ);
    extractPointsByIndices(boost::make_shared<pcl::PointIndices>(cluster_indices[0]),pts,pts_filtered,false,false);
    //pts.swap(pts_filtered);
    pcl::copyPointCloud(*pts_filtered,*pts);
}

void rgbdDetector::statisticalOutlierRemoval(PointCloudXYZ::Ptr pts, int num_neighbor,float stdDevMulThresh)
{
    PointCloudXYZ::Ptr pts_filtered(new PointCloudXYZ);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (pts);
    sor.setMeanK (num_neighbor);
    sor.setStddevMulThresh (stdDevMulThresh);
    sor.filter (*pts_filtered);
    //pts.swap(pts_filtered);
    pcl::copyPointCloud(*pts_filtered,*pts);
}

void rgbdDetector::voxelGridFilter(PointCloudXYZ::Ptr pts, float leaf_size)
{
    PointCloudXYZ::Ptr pts_filtered(new PointCloudXYZ);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pts);
    vg.setLeafSize(leaf_size,leaf_size,leaf_size);
    vg.filter(*pts_filtered);
    pcl::copyPointCloud(*pts_filtered,*pts);
}

void rgbdDetector::hypothesisVerification(vector<ClusterData>& cluster_data, float octree_res, float thresh, bool is_viz)
{
    vector<ClusterData>::iterator it =cluster_data.begin();
    for(;it!=cluster_data.end();)
    {
        pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(octree_res);
        octree.setInputCloud(it->scene_pc);
        octree.addPointsFromInputCloud();

        int count=0;

        std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> >::iterator iter_pc=it->model_pc->points.begin();
        for(;iter_pc!=it->model_pc->points.end();++iter_pc)
        {
            if(octree.isVoxelOccupiedAtPoint(*iter_pc))
                count++;
        }
        int model_pts=it->model_pc->points.size();

        if(is_viz)
        {
            pcl::visualization::PCLVisualizer v("check");
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
            v.addPointCloud(it->scene_pc,"scene");
            v.addPointCloud(it->model_pc,color,"model");
            v.spin();
            v.close();
        }

        double collision_rate = (double)count/(double)model_pts;
        if(collision_rate<thresh)
        {
//            if(it->matches.size()<15)
//            {
//                it=cluster_data.erase(it);
//            }
//            else
//            {
//                it++;
//            }
            it=cluster_data.erase(it);
        }
        else
            {
            it++;
        }

    }

}

void rgbdDetector::icpNonLinearPoseRefine(vector<ClusterData>& cluster_data,PointCloudXYZ::Ptr pc,IMAGE_WIDTH image_width,int bias_x,cv::Rect mask_rect)
{
    for(vector<ClusterData>::iterator it = cluster_data.begin();it!=cluster_data.end();++it)
    {
       //Get scene point cloud indices
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        indices=getPointCloudIndices(it,image_width,bias_x,mask_rect);
       //Extract scene pc according to indices
        PointCloudXYZ::Ptr scene_pc(new PointCloudXYZ);
        extractPointsByIndices(indices,pc,scene_pc,false,false);

        //Viz for test
        pcl::visualization::PCLVisualizer v("view_test");
        v.addPointCloud(scene_pc,"scene");
        pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(it->model_pc);
        v.addPointCloud(it->model_pc,color,"model");
        v.spin();

        //Remove Nan points
        vector<int> index;
        it->model_pc->is_dense=false;
        pcl::removeNaNFromPointCloud(*(it->model_pc),*(it->model_pc),index);
        pcl::removeNaNFromPointCloud(*scene_pc,*scene_pc,index);

        //Statistical outlier removal
        statisticalOutlierRemoval(scene_pc,50,1.0);

        //euclidianClustering(scene_pc,0.01);

//                 float leaf_size=0.002;
//                 voxelGridFilter(scene_pc,leaf_size);
//                 voxelGridFilter(it->model_pc,leaf_size);

        //Viz for test
        v.updatePointCloud(scene_pc,"scene");
        v.updatePointCloud(it->model_pc,color,"model");
        v.spin();

        //Instantiate a non-linear ICP object
        pcl::IterativeClosestPointNonLinear<pcl::PointXYZ,pcl::PointXYZ> icp_;
        icp_.setMaxCorrespondenceDistance(0.05);
        icp_.setMaximumIterations(50);
        icp_.setRANSACOutlierRejectionThreshold(0.02);
        icp_.setTransformationEpsilon (1e-8);
        icp_.setEuclideanFitnessEpsilon (0.002);

        //Coarse alignment
        icp_.setInputSource (it->model_pc);
        icp_.setInputTarget (scene_pc);
        icp_.align (*(it->model_pc));
        if(!icp_.hasConverged())
            cout<<"ICP cannot converge"<<endl;
        //Update pose
        Eigen::Matrix4f tf_mat = icp_.getFinalTransformation();
        Eigen::Matrix4d tf_mat_d=tf_mat.cast<double>();
        Eigen::Affine3d tf(tf_mat_d);
        it->pose=tf*it->pose;

        //Fine alignment 1
        icp_.setRANSACOutlierRejectionThreshold(0.01);
        icp_.setMaximumIterations(20);
        icp_.setMaxCorrespondenceDistance(0.02);
        icp_.setInputSource (it->model_pc);
        icp_.setInputTarget (scene_pc);
        icp_.align (*(it->model_pc));
        if(!icp_.hasConverged())
            cout<<"ICP cannot converge"<<endl;
        //Update pose
        tf_mat = icp_.getFinalTransformation();
        tf_mat_d=tf_mat.cast<double>();
        tf.matrix()=tf_mat_d;
        it->pose=tf*it->pose;

        //Fine alignment 2
        icp_.setMaximumIterations(10);
        icp_.setMaxCorrespondenceDistance(0.005);
        icp_.setInputSource (it->model_pc);
        icp_.setInputTarget (scene_pc);
        icp_.align (*(it->model_pc));
        if(!icp_.hasConverged())
            cout<<"ICP cannot converge"<<endl;
        //Update pose
        tf_mat = icp_.getFinalTransformation();
        tf_mat_d=tf_mat.cast<double>();
        tf.matrix()=tf_mat_d;
        it->pose=tf*it->pose;

        //Viz test
        v.updatePointCloud(it->model_pc,color,"model");
        v.spin();

    }
}


//Utility
pcl::PointIndices::Ptr rgbdDetector::getPointCloudIndices(vector<ClusterData>::iterator& it, IMAGE_WIDTH image_width,int bias_x,cv::Rect mask_rect)
{
    int x_cropped=0;
    int y_cropped=0;
    Mat cropped_mask=it->mask(mask_rect);
//    imshow("crop mask",cropped_mask);
//    imshow("uncrop mask",it->mask);
//    waitKey(0);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    for(int i=0;i<cropped_mask.rows;++i)
    {
        const uchar* row_data=cropped_mask.ptr<uchar>(i);
        for(int j=0;j<cropped_mask.cols;++j)
        {
            //Notice: the coordinate of input params "rect" is w.r.t cropped image, so the offset is needed to transform coordinate
            if(row_data[j]>0)
            {
                x_cropped=j+it->rect.x;
                y_cropped=i+it->rect.y;
                //Attention: image width of ensenso: 752, image height of ensenso: 480
                int index=y_cropped*image_width+x_cropped+bias_x;
                indices->indices.push_back(index);
            }
        }
    }
    return indices;

}

pcl::PointIndices::Ptr rgbdDetector::getPointCloudIndices(const cv::Rect& rect, IMAGE_WIDTH image_width,int bias_x)
{
    int row_offset=rect.y;
    int col_offset=rect.x;
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    for(int i=0;i<rect.height;++i)
    {
        for(int j=0;j<rect.width;++j)
        {
            //Notice: the coordinate of input params "rect" is w.r.t cropped image, so the offset is needed to transform coordinate
            int x_cropped=j+col_offset;
            int y_cropped=i+row_offset;
            int x_uncropped=x_cropped+bias_x;
            int y_uncropped=y_cropped;
            //Attention: image width of ensenso: 752, image height of ensenso: 480
            int index=y_uncropped*image_width+x_uncropped;
            indices->indices.push_back(index);

        }
    }
    return indices;

}

void rgbdDetector::extractPointsByIndices(pcl::PointIndices::Ptr indices, const PointCloudXYZ::Ptr ref_pts, PointCloudXYZ::Ptr extracted_pts, bool is_negative,bool is_organised)
{
    pcl::ExtractIndices<pcl::PointXYZ> tmp_extractor;
    tmp_extractor.setKeepOrganized(is_organised);
    tmp_extractor.setInputCloud(ref_pts);
    tmp_extractor.setNegative(is_negative);
    tmp_extractor.setIndices(indices);
    tmp_extractor.filter(*extracted_pts);
}


cv::Ptr<cv::linemod::Detector> rgbdDetector::readLinemod(const std::string& filename)
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
void rgbdDetector::readLinemodTemplateParams(const std::string fileName,
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

pointcloud_publisher::pointcloud_publisher(ros::NodeHandle& nh, const string &topic)
{
    publisher = nh.advertise<sensor_msgs::PointCloud2>(topic,1);    
    pc_msg.header.stamp = ros::Time::now();
}

void pointcloud_publisher::publish(PointCloudXYZ::Ptr pc)
{
    pcl::toROSMsg(*pc,pc_msg);
    pc_msg.header.frame_id = "/camera_link";
    publisher.publish(pc_msg);
}

void pointcloud_publisher::publish(sensor_msgs::PointCloud2& pc_msg_)
{
    pc_msg_.header.frame_id = "/camera_link";
    publisher.publish(pc_msg_);
}

void pointcloud_publisher::publish(PointCloudXYZ::Ptr pc,Eigen::Affine3d pose,const Scalar& color)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*pc,*pc_rgb);

    pcl::toROSMsg(*pc_rgb,pc_msg);
    pc_msg.header.frame_id = "/camera_link";

    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(pc_msg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(pc_msg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(pc_msg, "b");
    vector<pcl::PointXYZRGB,Eigen::aligned_allocator<pcl::PointXYZRGB> >::iterator it = pc_rgb->begin();

    for(;it!=pc_rgb->end();++it,++iter_r,++iter_g,++iter_b)
    {
        *iter_r = color[0];
        *iter_g = color[1];
        *iter_b = color[2];
    }

    publisher.publish(pc_msg);

    //TF
    tf::Transform pose_tf;
    tf::poseEigenToTF(pose,pose_tf);
    tf_broadcaster.sendTransform (tf::StampedTransform(pose_tf,ros::Time::now(),pc->header.frame_id,"object_frame"));

}



