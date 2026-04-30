#ifndef MYPCL_HPP
#define MYPCL_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vector_vec3d;
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > vector_quad;
// typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZ PointType;
// typedef pcl::PointXYZI PointType;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace mypcl
{
  struct pose
  {
    pose(Eigen::Quaterniond _q = Eigen::Quaterniond(1, 0, 0, 0),
         Eigen::Vector3d _t = Eigen::Vector3d(0, 0, 0)):q(_q), t(_t){}
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
  };

  void loadPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr& pc, int num,
               std::string prefix = "")
  {
    std::stringstream ss;
    if(pcd_fill_num > 0)
      ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
    else
      ss << num;
    pcl::io::loadPCDFile(filePath + prefix + ss.str() + ".pcd", *pc);
  }

  void savdPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr& pc, int num)
  {
    std::stringstream ss;
    if(pcd_fill_num > 0)
      ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
    else
      ss << num;
    pcl::io::savePCDFileBinary(filePath + ss.str() + ".pcd", *pc);
  }
  
  std::vector<pose> read_pose(std::string filename,
                              Eigen::Quaterniond qe = Eigen::Quaterniond(1, 0, 0, 0),
                              Eigen::Vector3d te = Eigen::Vector3d(0, 0, 0))
  {
    std::vector<pose> pose_vec;
    std::fstream file;
    file.open(filename);
    if(!file.is_open())
      throw std::runtime_error("Cannot open HBA pose file: " + filename);
    double tx, ty, tz, w, x, y, z;
    while(file >> tx >> ty >> tz >> w >> x >> y >> z)
    {
      Eigen::Quaterniond q(w, x, y, z);
      if(!std::isfinite(q.norm()) || q.norm() == 0.0)
        throw std::runtime_error("Invalid quaternion in HBA pose file: " + filename);
      q.normalize();
      Eigen::Vector3d t(tx, ty, tz);
      pose_vec.push_back(pose(qe * q, qe * t + te));
    }
    file.close();
    if(pose_vec.empty())
      throw std::runtime_error("No HBA poses loaded from: " + filename);
    std::cout << "loaded HBA poses: " << pose_vec.size() << std::endl;
    return pose_vec;
  }

  std::vector<pose> read_tum_pose(const std::string& filename,
                                  std::vector<std::string>& timestamps)
  {
    std::vector<pose> pose_vec;
    timestamps.clear();

    std::ifstream file(filename.c_str());
    if(!file.is_open())
      throw std::runtime_error("Cannot open TUM pose file: " + filename);

    std::string line;
    int line_num = 0;
    while(std::getline(file, line))
    {
      line_num++;
      size_t first = line.find_first_not_of(" \t\r\n");
      if(first == std::string::npos || line[first] == '#')
        continue;

      std::istringstream iss(line.substr(first));
      std::string timestamp;
      double tx, ty, tz, qx, qy, qz, qw;
      if(!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
        throw std::runtime_error("Malformed TUM pose line " + std::to_string(line_num) + " in: " + filename);

      Eigen::Quaterniond q(qw, qx, qy, qz);
      if(!std::isfinite(q.norm()) || q.norm() == 0.0)
        throw std::runtime_error("Invalid quaternion in TUM pose file: " + filename);
      q.normalize();
      pose_vec.push_back(pose(q, Eigen::Vector3d(tx, ty, tz)));
      timestamps.push_back(timestamp);
    }

    if(file.bad())
      throw std::runtime_error("Error while reading TUM pose file: " + filename);
    if(pose_vec.size() != timestamps.size())
      throw std::runtime_error("TUM timestamp/pose length mismatch in: " + filename);
    if(pose_vec.empty())
      throw std::runtime_error("No TUM poses loaded from: " + filename);

    std::cout << "loaded TUM poses: " << pose_vec.size() << std::endl;
    return pose_vec;
  }

  void transform_pointcloud(pcl::PointCloud<PointType> const& pc_in,
                            pcl::PointCloud<PointType>& pt_out,
                            Eigen::Vector3d t,
                            Eigen::Quaterniond q)
  {
    size_t size = pc_in.points.size();
    pt_out.points.resize(size);
    for(size_t i = 0; i < size; i++)
    {
      Eigen::Vector3d pt_cur(pc_in.points[i].x, pc_in.points[i].y, pc_in.points[i].z);
      Eigen::Vector3d pt_to;
      // if(pt_cur.norm()<0.3) continue;
      pt_to = q * pt_cur + t;
      pt_out.points[i].x = pt_to.x();
      pt_out.points[i].y = pt_to.y();
      pt_out.points[i].z = pt_to.z();
      // pt_out.points[i].r = pc_in.points[i].r;
      // pt_out.points[i].g = pc_in.points[i].g;
      // pt_out.points[i].b = pc_in.points[i].b;
    }
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr append_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc1,
                                                      pcl::PointCloud<pcl::PointXYZRGB> pc2)
  {
    size_t size1 = pc1->points.size();
    size_t size2 = pc2.points.size();
    pc1->points.resize(size1 + size2);
    for(size_t i = size1; i < size1 + size2; i++)
    {
      pc1->points[i].x = pc2.points[i-size1].x;
      pc1->points[i].y = pc2.points[i-size1].y;
      pc1->points[i].z = pc2.points[i-size1].z;
      pc1->points[i].r = pc2.points[i-size1].r;
      pc1->points[i].g = pc2.points[i-size1].g;
      pc1->points[i].b = pc2.points[i-size1].b;
      // pc1->points[i].intensity = pc2.points[i-size1].intensity;
    }
    return pc1;
  }

  pcl::PointCloud<PointType>::Ptr append_cloud(pcl::PointCloud<PointType>::Ptr pc1,
                                               pcl::PointCloud<PointType> pc2)
  {
    size_t size1 = pc1->points.size();
    size_t size2 = pc2.points.size();
    pc1->points.resize(size1 + size2);
    for(size_t i = size1; i < size1 + size2; i++)
    {
      pc1->points[i].x = pc2.points[i-size1].x;
      pc1->points[i].y = pc2.points[i-size1].y;
      pc1->points[i].z = pc2.points[i-size1].z;
      // pc1->points[i].r = pc2.points[i-size1].r;
      // pc1->points[i].g = pc2.points[i-size1].g;
      // pc1->points[i].b = pc2.points[i-size1].b;
      // pc1->points[i].intensity = pc2.points[i-size1].intensity;
    }
    return pc1;
  }

  double compute_inlier_ratio(std::vector<double> residuals, double ratio)
  {
    std::set<double> dis_vec;
    for(size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
      dis_vec.insert(fabs(residuals[3 * i + 0]) +
                     fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));

    return *(std::next(dis_vec.begin(), (int)((ratio) * dis_vec.size())));
  }

  void write_pose(std::vector<pose>& pose_vec, std::string path)
  {
    std::ofstream file;
    file.open(path + "pose.json", std::ofstream::trunc);
    file.close();
    Eigen::Quaterniond q0(pose_vec[0].q.w(), pose_vec[0].q.x(), pose_vec[0].q.y(), pose_vec[0].q.z());
    Eigen::Vector3d t0(pose_vec[0].t(0), pose_vec[0].t(1), pose_vec[0].t(2));
    file.open(path + "pose.json", std::ofstream::app);

    for(size_t i = 0; i < pose_vec.size(); i++)
    {
      pose_vec[i].t << q0.inverse()*(pose_vec[i].t-t0);
      pose_vec[i].q.w() = (q0.inverse()*pose_vec[i].q).w();
      pose_vec[i].q.x() = (q0.inverse()*pose_vec[i].q).x();
      pose_vec[i].q.y() = (q0.inverse()*pose_vec[i].q).y();
      pose_vec[i].q.z() = (q0.inverse()*pose_vec[i].q).z();
      file << pose_vec[i].t(0) << " "
           << pose_vec[i].t(1) << " "
           << pose_vec[i].t(2) << " "
           << pose_vec[i].q.w() << " " << pose_vec[i].q.x() << " "
           << pose_vec[i].q.y() << " " << pose_vec[i].q.z();
      if(i < pose_vec.size()-1) file << "\n";
    }
    file.close();
  }

  void writeEVOPose(std::vector<double>& lidar_times, std::vector<pose>& pose_vec, std::string path)
  {
    std::ofstream file;
    file.open(path + "evo_pose.txt", std::ofstream::trunc);
    for(size_t i = 0; i < pose_vec.size(); i++)
    {
      file << std::setprecision(18) << lidar_times[i] << " " << std::setprecision(6)
           << pose_vec[i].t(0) << " " << pose_vec[i].t(1) << " " << pose_vec[i].t(2) << " "
           << pose_vec[i].q.x() << " " << pose_vec[i].q.y() << " "
           << pose_vec[i].q.z() << " " << pose_vec[i].q.w();
      if(i < pose_vec.size()-1) file << "\n";
    }
    file.close();
  }

  void write_tum_pose(const std::vector<std::string>& timestamps,
                      const std::vector<pose>& pose_vec,
                      const std::string& filename)
  {
    if(timestamps.size() != pose_vec.size())
      throw std::runtime_error("TUM output timestamp/pose length mismatch");

    std::ofstream file(filename.c_str(), std::ofstream::trunc);
    if(!file.is_open())
      throw std::runtime_error("Cannot open TUM output pose file: " + filename);

    for(size_t i = 0; i < pose_vec.size(); i++)
    {
      Eigen::Quaterniond q = pose_vec[i].q;
      if(!std::isfinite(q.norm()) || q.norm() == 0.0)
        throw std::runtime_error("Invalid optimized quaternion while writing TUM poses");
      q.normalize();

      file << timestamps[i] << " "
           << std::setprecision(9)
           << pose_vec[i].t(0) << " " << pose_vec[i].t(1) << " " << pose_vec[i].t(2) << " "
           << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
      if(i < pose_vec.size()-1) file << "\n";
    }
    file.close();
    std::cout << "saved TUM poses: " << pose_vec.size() << std::endl;
  }
}

#endif
