/*
Copyright (c) 2019-2020, Juan Miguel Jimeno
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef QUADRUPED_CONTROLLER_H
#define QUADRUPED_CONTROLLER_H

#include "ros/ros.h"

#include "libtorch.hpp"

#include <champ_msgs/Joints.h>
#include <champ_msgs/Pose.h>
#include <champ_msgs/PointArray.h>
#include <champ_msgs/ContactsStamped.h>

#include <champ/utils/urdf_loader.h>
#include <champ/body_controller/body_controller.h>
#include <champ/leg_controller/leg_controller.h>
#include <champ/kinematics/kinematics.h>

#include "tf/transform_datatypes.h"
#include <gazebo_msgs/ContactsState.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/SetModelState.h>
#include <std_srvs/Empty.h>
#include <std_srvs/Trigger.h>

class QuadrupedController
{
    ros::Subscriber cmd_vel_subscriber_;
    ros::Subscriber cmd_pose_subscriber_;
    
    ros::Publisher joint_states_publisher_;   
    ros::Publisher joint_commands_publisher_;   
    ros::Publisher foot_contacts_publisher_;

    ros::Timer loop_timer_;

    champ::Velocities req_vel_;
    champ::Pose req_pose_;

    champ::GaitConfig gait_config_;

    // vvv For RL locomotion controller

    // methods
    void controlLoopRl_(const ros::TimerEvent& event);
    void updateGaitParameters_();
    void getStateForRlLocoController_();
    void getActionFromRlLocoController_();
    bool selfRightRobotSrvCallback_(std_srvs::Trigger::Request& req,
                                    std_srvs::Trigger::Response& res);
    void setRobotUpright_();

    void setJointControllersGains_();

    inline double wrap_01(double a) {
      return a - fastFloor(a);
    }

    inline int fastFloor(double a) {
      int i = int(a);
      if (i > a)
        i--;

      return i;
    }

    inline Eigen::Matrix<double, 4, 4>
    getTransformBtoIMU(double pos_offset[3], double orientation_offset[3]) {
      Eigen::Matrix<double, 4, 4> T_B_to_imu;
      T_B_to_imu(0, 3) = pos_offset[0];
      T_B_to_imu(1, 3) = pos_offset[1];
      T_B_to_imu(2, 3) = pos_offset[2];
      T_B_to_imu(3, 3) = 1.;

      Eigen::Matrix<double, 3, 3> mat;
      mat = getEulerRotMatFromRPY(orientation_offset);

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          T_B_to_imu(i, j) = mat(i, j);
        }
      }

      return T_B_to_imu;
    }

    inline void
    getRotationAndOffsetFromTransformation(Eigen::Matrix<double, 4, 4> T,
                                           Eigen::Matrix<double, 3, 3>& rot,
                                           Eigen::Vector3d& offset) {
      for (int i = 0; i < 3; i++) {
        offset(i) = T(i, 3);
        for (int j = 0; j < 3; j++) {
          rot(i, j) = T(i, j);
        }
      }

      return;
    }

    inline Eigen::Matrix<double, 3, 3> getEulerRotMatFromRPY(double angles[3]) {
      double psi = angles[0];
      double theta = angles[1];
      double phi = angles[2];
      Eigen::Matrix<double, 3, 3> mat;
      mat(0, 0) = std::cos(theta) * std::cos(phi);
      mat(0, 1) = std::sin(psi) * std::sin(theta) * std::cos(phi) -
          std::cos(psi) * std::sin(phi);
      mat(0, 2) = std::cos(psi) * std::sin(theta) * std::cos(phi) +
          std::sin(psi) * std::sin(phi);
      mat(1, 0) = std::cos(theta) * std::sin(phi);
      mat(1, 1) = std::sin(psi) * std::sin(theta) * std::sin(phi) +
          std::cos(psi) * std::cos(phi);
      mat(1, 2) = std::cos(psi) * std::sin(theta) * std::sin(phi) -
          std::sin(psi) * std::cos(phi);
      mat(2, 0) = -std::sin(theta);
      mat(2, 1) = std::sin(psi) * std::cos(theta);
      mat(2, 2) = std::cos(psi) * std::cos(theta);

      return mat;
    }

    inline Eigen::Matrix<double, 3, 3> quatToRotMat(Eigen::Vector4d q) {
      // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
      Eigen::Matrix<double, 3, 3> R;
      R << q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3),
          2 * q(1) * q(2) - 2 * q(0) * q(3), 2 * q(0) * q(2) + 2 * q(1) * q(3),
          2 * q(0) * q(3) + 2 * q(1) * q(2),
          q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3),
          2 * q(2) * q(3) - 2 * q(0) * q(1), 2 * q(1) * q(3) - 2 * q(0) * q(2),
          2 * q(0) * q(1) + 2 * q(2) * q(3),
          q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3);

      return R;
    }

    inline double clamp(double a, double min, double max) {
      return a > max ? max : (a < min ? min : a);
    }

    inline Eigen::Vector4d rotMatToQuat(Eigen::Matrix<double, 3, 3> R) {
      Eigen::Vector4d quat;
      quat(0) = sqrt(1 + R(0, 0) + R(1, 1) + R(2, 2)) / 2;
      quat(1) = (R(2, 1) - R(1, 2)) / (4 * quat(0));
      quat(2) = (R(0, 2) - R(2, 0)) / (4 * quat(0));
      quat(3) = (R(1, 0) - R(0, 1)) / (4 * quat(0));

      return quat;
    }

    inline Eigen::Vector4d quatInv(Eigen::Vector4d quat) {
      Eigen::Vector4d q_out;
      q_out(0) = quat(0);
      q_out(1) = -quat(1);
      q_out(2) = -quat(2);
      q_out(3) = -quat(3);

      return q_out;
    }

    inline Eigen::Vector4d quatMult(Eigen::Vector4d quat1,
                                    Eigen::Vector4d quat2) {
      // Note that the order of quaternion multiplication matters!
      Eigen::Vector4d quat;
      quat(0) = quat1(0) * quat2(0) - quat1(1) * quat2(1) -
          quat1(2) * quat2(2) - quat1(3) * quat2(3);
      quat(1) = quat1(0) * quat2(1) + quat1(1) * quat2(0) +
          quat1(2) * quat2(3) - quat1(3) * quat2(2);
      quat(2) = quat1(0) * quat2(2) - quat1(1) * quat2(3) +
          quat1(2) * quat2(0) + quat1(3) * quat2(1);
      quat(3) = quat1(0) * quat2(3) + quat1(1) * quat2(2) -
          quat1(2) * quat2(1) + quat1(3) * quat2(0);

      return quat;
    }

    void jointStatesCallback_(const sensor_msgs::JointState::ConstPtr& msg);
    void bodyVelocityCallback_(const geometry_msgs::Twist::ConstPtr& msg);
    void endEffectorContactCallback_(
        const gazebo_msgs::ContactsState::ConstPtr& msg);
    void imuCallback_(const sensor_msgs::Imu::ConstPtr& msg);
    void groundTruthOdomCallback_(const nav_msgs::OdometryConstPtr& msg);

    // members
    champ::RLControllerConfig rl_controller_config_;
    Eigen::Matrix<double, 362, 1> state_scaled_;
    Eigen::Matrix<double, 362, 1> state_unscaled_;
    Eigen::Matrix<double, 13, 1> action_;
    Eigen::VectorXd action_history_;
    Eigen::Vector4d quat_base_to_inertial_; ///< base quaternion
    Eigen::Vector4d quat_base_to_imu_;
    Eigen::Vector3d offset_base_to_imu_;
    Eigen::Matrix<double, 3, 3> rot_base_to_imu_;
    Eigen::VectorXd joint_velocity_;
    Eigen::VectorXd joint_position_;
    Eigen::Vector3d body_lin_velocity_;
    Eigen::Vector3d body_ang_velocity_;
    Eigen::Vector3d body_position_;
    Eigen::VectorXd currErr_;
    Eigen::VectorXd prevErr_;
    double phase_time_left_[4] = {1., 1., 1., 1.};
    bool foot_contacts_[4] = {false, false, false, false};
    bool desired_foot_contacts_[4] = {true, true, true, true};
    ;
    double target_foot_clearance_[4] = {0., 0., 0., 0.};
    bool is_stance_gait_ = false;
    double phase_ = 0.;
    int step_ = 0;
    std::string imu_topic_name_;
    double imu_lin_offset_pos_[3] = {0., 0., 0.};
    double imu_ang_offset_pos_[3] = {0., 0., 0.};
    Eigen::Matrix<double, 4, 4> T_B_to_imu_;
    double control_dt_ = 0.024;
    double loop_rate_ = 0.004;
    int control_step_ = -1;
    gazebo_msgs::ModelState pose_;
    Eigen::Matrix<double, 3, 3> rotMatBtoI_;
    ros::ServiceClient set_spot_pose_srv_;
    ros::ServiceClient pause_gazebo_;
    ros::ServiceClient unpause_gazebo_;
    bool automatic_self_righting_enabled_ = false;
    std::string gazebo_model_name_;
    ros::ServiceServer self_righting_srv_;
    ros::WallTime start_;

    libtorch::libtorch net_; ///< libtorch object to hold network parametersc

    // subscribers
    ros::Subscriber body_velocity_subscriber_;
    ros::Subscriber end_effector_subscriber_;
    ros::Subscriber body_pos_subscriber_;
    ros::Subscriber joint_angle_subscriber_;
    ros::Subscriber ground_truth_odom_subscriber_;
    // message_filters::Subscriber<sensor_msgs::JointState>
    // joint_angle_subscriber_; message_filters::Subscriber<nav_msgs::Odometry>
    // ground_truth_odom_subscriber_;

    // typedef
    // message_filters::sync_policies::ApproximateTime<sensor_msgs::JointState,
    // nav_msgs::Odometry> rl_control_sync_policy_;
    // message_filters::Synchronizer<rl_control_sync_policy_> rl_control_sync_;

    void callback(const sensor_msgs::JointState::ConstPtr& joint_msg,
                  const nav_msgs::OdometryConstPtr& ground_truth_msg);

    //^^^

    champ::QuadrupedBase base_;
    champ::BodyController body_controller_;
    champ::LegController leg_controller_;
    champ::Kinematics kinematics_;

    std::vector<std::string> joint_names_;

    bool publish_foot_contacts_;
    bool publish_joint_states_;
    bool publish_joint_control_;
    bool in_gazebo_;

    void controlLoop_(const ros::TimerEvent& event);
    
    void publishJoints_(float target_joints[12]);
    void publishFootContacts_(bool foot_contacts[4]);

    void cmdVelCallback_(const geometry_msgs::Twist::ConstPtr& msg);
    void cmdPoseCallback_(const geometry_msgs::Pose::ConstPtr& msg);

    public:
        QuadrupedController(ros::NodeHandle *nh, ros::NodeHandle *pnh);
};

#endif