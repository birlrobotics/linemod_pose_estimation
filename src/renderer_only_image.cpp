//ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
//#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd/rgbd.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include <object_recognition_renderer/renderer3d.h>
#include <object_recognition_renderer/utils.h>

// Functions to store detector and templates in single XML/YAML file
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

static void writeLinemod(const cv::Ptr<cv::linemod::Detector>& detector, const std::string& filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  detector->write(fs);

  std::vector<cv::String> ids = detector->classIds();
  fs << "classes" << "[";
  for (int i = 0; i < (int)ids.size(); ++i)
  {
    fs << "{";
    detector->writeClass(ids[i], fs);
    fs << "}"; // current class
  }
  fs << "]"; // classes
}

static void writeLinemodTemplateParams(std::string fileName,
                                       std::vector<cv::Mat>& Rs,
                                       std::vector<cv::Mat>& Ts,
                                       std::vector<double>& distances,
                                       std::vector<double>& obj_origin_dists,
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
    cv::FileStorage fs(fileName,cv::FileStorage::WRITE);
    int num_templates=Rs.size ();
    for(int i=0;i<num_templates;++i)
    {
        std::stringstream ss;
        std::string a;
        ss<<i;
        a="Template ";
        a+=ss.str ();
        fs<<a<<"{";
        fs<<"ID"<<i;
        fs<<"R"<<Rs[i];
        fs<<"T"<<Ts[i];
        fs<<"K"<<Ks[i];
        fs<<"D"<<distances[i];
        fs<<"Ori_dist"<<obj_origin_dists[i];
        fs<<"Rect"<<Rects[i];
        fs<<"}";
    }
    //fs<<"K Intrinsic Matrix"<<cv::Mat(K_matrix);
    fs<<"renderer_n_points"<<renderer_n_points;
    fs<<"renderer_angle_step"<<renderer_angle_step;
    fs<<"renderer_radius_min"<<renderer_radius_min;
    fs<<"renderer_radius_max"<<renderer_radius_max;
    fs<<"renderer_radius_step"<<renderer_radius_step;
    fs<<"renderer_width"<<renderer_width;
    fs<<"renderer_height"<<renderer_height;
    fs<<"renderer_focal_length_x"<<renderer_focal_length_x;
    fs<<"renderer_focal_length_y"<<renderer_focal_length_y;
    fs<<"renderer_near"<<renderer_near;
    fs<<"renderer_far"<<renderer_far;
    fs.release ();
}

static void writeLinemodRender(std::string& fileName,
                                       std::vector<cv::Mat>& depth_img,
                                       std::vector<cv::Mat>& mask,
                                       std::vector<cv::Rect>& rect)
{
    cv::FileStorage fs(fileName,cv::FileStorage::WRITE);
    int num_templates=depth_img.size ();
    for(int i=0;i<num_templates;++i)
    {
        std::stringstream ss;
        std::string a;
        ss<<i;
        a="Template ";
        a+=ss.str ();
        fs<<a<<"{";
        fs<<"ID"<<i;
        fs<<"Depth"<<depth_img[i];
        fs<<"Mask"<<mask[i];
        fs<<"Rect"<<rect[i];
        fs<<"}";
    }
    fs.release ();
}

int main(int argc,char** argv)
{
    std::vector< cv::Ptr<cv::linemod::Modality> > modalities;
    //cv::Ptr<cv::linemod::ColorGradient> color_grad(new cv::linemod::ColorGradient);
    //cv::Ptr<cv::linemod::DepthNormal> depth_grad(new cv::linemod::DepthNormal);
    modalities.push_back(cv::Ptr<cv::linemod::ColorGradient>(new cv::linemod::ColorGradient));
    std::vector<int> ensenso_T;
    ensenso_T.push_back(5);
    ensenso_T.push_back(8);
    cv::Ptr<cv::linemod::Detector> detector_(new cv::linemod::Detector(modalities,ensenso_T));

    int renderer_n_points_;
    int renderer_angle_step_;
    double renderer_radius_min_;
    double renderer_radius_max_;
    double renderer_radius_step_;
    int renderer_width_;
    int renderer_height_;
    double renderer_near_;
    double renderer_far_ ;
    double renderer_focal_length_x_;//Kinect ;//xtion 570.342;
    double renderer_focal_length_y_;//Kinect //xtion 570.342;
    std::string stl_file;
    std::string template_output_path;
    std::string renderer_params_output_path;


    if(argc<8)
    {
        renderer_n_points_ = 270;     // it most maybe is plane average view points
        renderer_angle_step_ = 5;     // (100-(-100))/3, it not influenced by is_restricted
        renderer_radius_min_ = 0.45;
        renderer_radius_max_ = 0.65;
        renderer_radius_step_ = 0.035;
        renderer_width_ = 752;
        renderer_height_ = 480;
        renderer_near_ = 0.1;
        renderer_far_ = 1000.0;
        renderer_focal_length_x_ = 826.119324;//Kinect ;//carmine 535.566011; //dataset 571.9737
        renderer_focal_length_y_ = 826.119324;//Kinect //carmine 537.168115;  //dataset 571.0073
        stl_file="/home/birl-spai-ubuntu14/ur_ws/src/linemod_pose_estimation/config/stl/memoryChip2.stl";
        template_output_path="/home/birl-spai-ubuntu14/ur_ws/src/linemod_pose_estimation/config/data/memoryChip2_0_ensenso_templates.yml";
        renderer_params_output_path="/home/birl-spai-ubuntu14/ur_ws/src/linemod_pose_estimation/config/data/memoryChip2_0_ensenso_renderer_params.yml";

    }
    else
    {
     renderer_n_points_ = 150;
     renderer_angle_step_ = 10;
     renderer_radius_min_ = 0.4;
     renderer_radius_max_ = 0.8;
     renderer_radius_step_ = 0.2;
     renderer_width_ = atoi(argv[3]);
     renderer_height_ = atoi(argv[4]);
     renderer_near_ = 0.1;
     renderer_far_ = 1000.0;
     renderer_focal_length_x_ = atof(argv[1]);//Kinect ;//xtion 570.342;
     renderer_focal_length_y_ = atof(argv[2]);//Kinect //xtion 570.342;
     stl_file=argv[5];
     template_output_path=argv[6];
     renderer_params_output_path=argv[7];
    }

    Renderer3d render=Renderer3d(stl_file);
    render.set_parameters (renderer_width_, renderer_height_, renderer_focal_length_x_,
                           renderer_focal_length_y_, renderer_near_, renderer_far_);
    RendererIterator renderer_iterator=RendererIterator(&render,renderer_n_points_);
    renderer_iterator.angle_step_ = renderer_angle_step_;
    renderer_iterator.radius_min_ = float(renderer_radius_min_);
    renderer_iterator.radius_max_ = float(renderer_radius_max_);
    renderer_iterator.radius_step_ = float(renderer_radius_step_);

    cv::Mat image, depth, mask,flip_mask,flip_depth,flip_image;;
    cv::Matx33d R_cam,R;
    cv::Vec3d T;
    cv::Matx33f K;
    std::vector<cv::Mat> Rs_;
    std::vector<cv::Mat> Ts_;
    std::vector<cv::Rect> Rects_;
    std::vector<double> distances_;
    std::vector<double> Origin_dists_;
    std::vector<cv::Mat> Ks_;
    std::vector<cv::Mat> depth_img;
    std::vector<cv::Mat> masks;
    std::vector<cv::Rect> rects;

    for (size_t i = 0; !renderer_iterator.isDone(); ++i, ++renderer_iterator)
    {
      std::stringstream status;
      status << "Loading images " << (i+1) << "/"
          << renderer_iterator.n_templates();
      std::cout << status.str();

      cv::Rect rect;
      bool is_restricted=true; //Whether the viewport will be restricted for plannar object
      bool is_image_valid;
      renderer_iterator.render(image, depth, mask, rect,is_restricted,is_image_valid);
      cv::flip(mask,flip_mask,0);
      cv::flip(depth,flip_depth,0);
      cv::flip(image,flip_image,0);

      if(is_image_valid)
      {
          R_cam = renderer_iterator.R_cam();
          R = R_cam.inv();                              // R means in camera coordinate
          T = renderer_iterator.T();                    //
          //D_obj distance from camera to object origin ; it means in camera coordinate ; and it is translation`s Z=D_obj, X=Y=0;
          double surface_center=double(depth.at<ushort>(depth.rows/2.0f, depth.cols/2.0f)/1000.0f);
          double distance;
          if(surface_center==0)
          {
            distance = (double)renderer_iterator.D_obj() - surface_center;
          }else
          {
            distance = (double)renderer_iterator.D_obj() - surface_center;
          }
          double obj_origin_dist=(double)renderer_iterator.D_obj();
          std::cout<<"obj_origin_dist"<<obj_origin_dist;
          K = cv::Matx33f(float(renderer_focal_length_x_), 0.0f, float(flip_image.cols)/2.0f, 0.0f, float(renderer_focal_length_y_), float(flip_image.rows)/2.0f, 0.0f, 0.0f, 1.0f);

          std::vector<cv::Mat> sources(1);
          sources[0] = flip_image;


          // Display the rendered image
          if (true)
          {
            cv::namedWindow("Rendering RGB");
            cv::namedWindow("Rendering Depth");
            cv::namedWindow("Rendering Mask");
            if (!image.empty()) {
              cv::imshow("Rendering RGB", image);
              cv::imshow("Rendering Depth", depth);
              cv::imshow("Rendering Mask", mask);
              cv::waitKey(1);
            }
          }


          int template_in = detector_->addTemplate(sources, "obj", flip_mask);
          if (template_in == -1)
          {
            // Delete the status
            for (size_t j = 0; j < status.str().size(); ++j)
              std::cout << '\b';
            continue;
          }

          // Also store the pose of each template
          Rs_.push_back(cv::Mat(R));
          Ts_.push_back(cv::Mat(T));
          distances_.push_back(distance);
          Ks_.push_back(cv::Mat(K));
          Rects_.push_back(rect);
          Origin_dists_.push_back (obj_origin_dist);

          //Store depth image, mask and rect
          //      depth_img.push_back(depth);
          //      masks.push_back(mask);
          //      rects.push_back(rect);

          // Delete the status
          for (size_t j = 0; j < status.str().size(); ++j)
            std::cout << '\b';
      }

    }


    writeLinemod (detector_,template_output_path);
    writeLinemodTemplateParams (renderer_params_output_path,
                                Rs_,
                                Ts_,
                                distances_,
                                Origin_dists_,
                                Ks_,
                                Rects_,
                                renderer_n_points_,
                                renderer_angle_step_,
                                renderer_radius_min_,
                                renderer_radius_max_,
                                renderer_radius_step_,
                                renderer_width_,
                                renderer_height_,
                                renderer_focal_length_x_,
                                renderer_focal_length_y_,
                                renderer_near_,
                                renderer_far_);
    //writeLinemodRender(renderer_depth_output_path,depth_img,masks,rects);

      return 0;

}
