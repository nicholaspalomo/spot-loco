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

#include <quadruped_controller.h>

QuadrupedController::QuadrupedController(ros::NodeHandle* nh,
                                         ros::NodeHandle* pnh)
  : body_controller_(base_),
    leg_controller_(base_),
    kinematics_(base_)
{
  std::string joint_control_topic = "joint_group_position_controller/command";
  std::string knee_orientation;
  double loop_rate = 250.0;

  nh->getParam("gait/pantograph_leg", gait_config_.pantograph_leg);
  nh->getParam("gait/max_linear_velocity_x",
               gait_config_.max_linear_velocity_x);
  nh->getParam("gait/max_linear_velocity_y",
               gait_config_.max_linear_velocity_y);
  nh->getParam("gait/max_angular_velocity_z",
               gait_config_.max_angular_velocity_z);
  nh->getParam("gait/com_x_translation", gait_config_.com_x_translation);
  nh->getParam("gait/swing_height", gait_config_.swing_height);
  nh->getParam("gait/stance_depth", gait_config_.stance_depth);
  nh->getParam("gait/stance_duration", gait_config_.stance_duration);
  nh->getParam("gait/nominal_height", gait_config_.nominal_height);
  nh->getParam("gait/knee_orientation", knee_orientation);
  pnh->getParam("publish_foot_contacts", publish_foot_contacts_);
  pnh->getParam("publish_joint_states", publish_joint_states_);
  pnh->getParam("publish_joint_control", publish_joint_control_);
  pnh->getParam("gazebo", in_gazebo_);
  pnh->getParam("joint_controller_topic", joint_control_topic);
  pnh->getParam("loop_rate", loop_rate);

  // vvv For RL locomotion controller

  // Load the parameters for the RL controller from the parameter server
  nh->getParam("rl_controller/p_gain", rl_controller_config_.p_gain);
  nh->getParam("rl_controller/d_gain", rl_controller_config_.d_gain);
  nh->getParam("rl_controller/control_dt", rl_controller_config_.control_dt);
  nh->getParam("rl_controller/obs_dim", rl_controller_config_.obs_dim);
  nh->getParam("rl_controller/acts_dim", rl_controller_config_.acts_dim);
  nh->getParam("rl_controller/target_vel_lim_x",
               rl_controller_config_.target_vel_lim_x);
  nh->getParam("rl_controller/target_vel_lim_y",
               rl_controller_config_.target_vel_lim_y);
  nh->getParam("rl_controller/target_vel_lim_r",
               rl_controller_config_.target_vel_lim_r);
  nh->getParam("rl_controller/n_joints", rl_controller_config_.n_joints);

  // Load action scalings from parameter server
  nh->getParam("rl_controller/action_scaling/action_mean/haa",
               rl_controller_config_.action_mean_haa);
  nh->getParam("rl_controller/action_scaling/action_mean/hfe",
               rl_controller_config_.action_mean_hfe);
  nh->getParam("rl_controller/action_scaling/action_mean/kfe",
               rl_controller_config_.action_mean_kfe);
  nh->getParam("rl_controller/action_scaling/action_mean/gait_frequency",
               rl_controller_config_.action_mean_gait_freq);
  nh->getParam("rl_controller/action_scaling/action_std/haa",
               rl_controller_config_.action_std_haa);
  nh->getParam("rl_controller/action_scaling/action_std/hfe",
               rl_controller_config_.action_std_hfe);
  nh->getParam("rl_controller/action_scaling/action_std/kfe",
               rl_controller_config_.action_std_kfe);
  nh->getParam("rl_controller/action_scaling/action_std/gait_frequency",
               rl_controller_config_.action_std_gait_freq);

  // Load observation scalings from parameter server
  nh->getParam("rl_controller/observation_scaling/observation_mean/base_height",
               rl_controller_config_.obs_mean_base_height);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/gravity_axis",
      rl_controller_config_.obs_mean_grav_axis);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/joint_position/haa",
      rl_controller_config_.obs_mean_joint_pos_haa);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/joint_position/hfe",
      rl_controller_config_.obs_mean_joint_pos_hfe);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/joint_position/kfe",
      rl_controller_config_.obs_mean_joint_pos_kfe);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/body_lin_vel",
      rl_controller_config_.obs_mean_body_lin_vel);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/body_ang_vel",
      rl_controller_config_.obs_mean_body_ang_vel);
  nh->getParam("rl_controller/observation_scaling/observation_mean/joint_vel",
               rl_controller_config_.obs_mean_joint_vel);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/target_vel/x",
      rl_controller_config_.obs_mean_target_vel_x);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/target_vel/y",
      rl_controller_config_.obs_mean_target_vel_y);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/target_vel/r",
      rl_controller_config_.obs_mean_target_vel_r);
  nh->getParam("rl_controller/observation_scaling/observation_mean/action_hist",
               rl_controller_config_.obs_mean_action_hist);
  nh->getParam("rl_controller/observation_scaling/observation_mean/stride",
               rl_controller_config_.obs_mean_stride);
  nh->getParam("rl_controller/observation_scaling/observation_mean/swing_start",
               rl_controller_config_.obs_mean_swing_start);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/swing_duration",
      rl_controller_config_.obs_mean_swing_duration);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/phase_time_left",
      rl_controller_config_.obs_mean_phase_time_left);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/foot_contact_state",
      rl_controller_config_.obs_mean_foot_contact);
  nh->getParam("rl_controller/observation_scaling/observation_mean/"
               "desired_contact_state",
               rl_controller_config_.obs_mean_des_foot_contact);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/ee_clearance",
      rl_controller_config_.obs_mean_ee_clearance);
  nh->getParam("rl_controller/observation_scaling/observation_mean/ee_target",
               rl_controller_config_.obs_mean_ee_target);
  nh->getParam(
      "rl_controller/observation_scaling/observation_mean/body_vel_error",
      rl_controller_config_.obs_mean_body_vel_error);

  nh->getParam("rl_controller/observation_scaling/observation_std/base_height",
               rl_controller_config_.obs_std_base_height);
  nh->getParam("rl_controller/observation_scaling/observation_std/gravity_axis",
               rl_controller_config_.obs_std_grav_axis);
  nh->getParam(
      "rl_controller/observation_scaling/observation_std/joint_position/haa",
      rl_controller_config_.obs_std_joint_pos_haa);
  nh->getParam(
      "rl_controller/observation_scaling/observation_std/joint_position/hfe",
      rl_controller_config_.obs_std_joint_pos_hfe);
  nh->getParam(
      "rl_controller/observation_scaling/observation_std/joint_position/kfe",
      rl_controller_config_.obs_std_joint_pos_kfe);
  nh->getParam("rl_controller/observation_scaling/observation_std/body_lin_vel",
               rl_controller_config_.obs_std_body_lin_vel);
  nh->getParam("rl_controller/observation_scaling/observation_std/body_ang_vel",
               rl_controller_config_.obs_std_body_ang_vel);
  nh->getParam("rl_controller/observation_scaling/observation_std/joint_vel",
               rl_controller_config_.obs_std_joint_vel);
  nh->getParam("rl_controller/observation_scaling/observation_std/target_vel/x",
               rl_controller_config_.obs_std_target_vel_x);
  nh->getParam("rl_controller/observation_scaling/observation_std/target_vel/y",
               rl_controller_config_.obs_std_target_vel_y);
  nh->getParam("rl_controller/observation_scaling/observation_std/target_vel/r",
               rl_controller_config_.obs_std_target_vel_r);
  nh->getParam("rl_controller/observation_scaling/observation_std/action_hist",
               rl_controller_config_.obs_std_action_hist);
  nh->getParam("rl_controller/observation_scaling/observation_std/stride",
               rl_controller_config_.obs_std_stride);
  nh->getParam("rl_controller/observation_scaling/observation_std/swing_start",
               rl_controller_config_.obs_std_swing_start);
  nh->getParam(
      "rl_controller/observation_scaling/observation_std/swing_duration",
      rl_controller_config_.obs_std_swing_duration);
  nh->getParam(
      "rl_controller/observation_scaling/observation_std/phase_time_left",
      rl_controller_config_.obs_std_phase_time_left);
  nh->getParam(
      "rl_controller/observation_scaling/observation_std/foot_contact_state",
      rl_controller_config_.obs_std_foot_contact);
  nh->getParam("rl_controller/observation_scaling/observation_std/"
               "desired_contact_state",
               rl_controller_config_.obs_std_des_foot_contact);
  nh->getParam("rl_controller/observation_scaling/observation_std/ee_clearance",
               rl_controller_config_.obs_std_ee_clearance);
  nh->getParam("rl_controller/observation_scaling/observation_std/ee_target",
               rl_controller_config_.obs_std_ee_target);
  nh->getParam(
      "rl_controller/observation_scaling/observation_std/body_vel_error",
      rl_controller_config_.obs_std_body_vel_error);

  nh->getParam("rl_controller/gait_params/stride",
               rl_controller_config_.gait_stride);
  nh->getParam("rl_controller/gait_params/max_foot_height",
               rl_controller_config_.max_foot_height);
  nh->getParam("rl_controller/gait_params/swing_start",
               *(rl_controller_config_.swing_start));
  nh->getParam("rl_controller/gait_params/swing_duration",
               *(rl_controller_config_.swing_duration));

  nh->getParam("rl_controller/sim_dt", loop_rate_);
  nh->getParam("rl_controller/control_dt", control_dt_);

  // Load the parameters for self-righting
  nh->getParam("rl_controller/self_right_params/automatic",
               automatic_self_righting_enabled_);
  nh->getParam("rl_controller/self_right_params/model_name",
               gazebo_model_name_);

  // Load the parameters for the IMU subscription
  nh->getParam("rl_controller/imu/topic_name", imu_topic_name_);
  nh->getParam("rl_controller/imu/position", *(imu_lin_offset_pos_));
  nh->getParam("rl_controller/imu/orientation", *(imu_ang_offset_pos_));

  // Get the transformation matrix from the IMU to the body center.
  T_B_to_imu_ = getTransformBtoIMU(imu_lin_offset_pos_, imu_ang_offset_pos_);
  getRotationAndOffsetFromTransformation(
      T_B_to_imu_, rot_base_to_imu_, offset_base_to_imu_);
  quat_base_to_imu_ = rotMatToQuat(rot_base_to_imu_);

  rl_controller_config_.initialize_controller_scalings();

  state_scaled_.setZero(rl_controller_config_.obs_dim);
  state_unscaled_.setZero(rl_controller_config_.obs_dim);
  joint_velocity_.setZero(rl_controller_config_.n_joints);
  joint_position_.setZero(rl_controller_config_.n_joints);
  action_history_.setZero(3 * rl_controller_config_.acts_dim);
  action_.setZero(rl_controller_config_.acts_dim);

  currErr_.setZero(rl_controller_config_.n_joints);
  prevErr_.setZero(rl_controller_config_.n_joints);

  joint_angle_subscriber_ = nh->subscribe(
      "joint_states", 100, &QuadrupedController::jointStatesCallback_, this);

  ground_truth_odom_subscriber_ =
      nh->subscribe("ground_truth/odometry",
                    100,
                    &QuadrupedController::groundTruthOdomCallback_,
                    this);

  set_spot_pose_srv_ =
      nh->serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
  self_righting_srv_ = nh->advertiseService(
      "self_right", &QuadrupedController::selfRightRobotSrvCallback_, this);
  pause_gazebo_ = nh->serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
  unpause_gazebo_ =
      nh->serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");

  std::string torchscriptNetDir(NETWORK_WEIGHTS_DIR);
  torchscriptNetDir.append("loco_controller.pt");
  net_.load(torchscriptNetDir);

  //^^^

  cmd_vel_subscriber_ = nh->subscribe(
      "cmd_vel/smooth", 100, &QuadrupedController::cmdVelCallback_, this);
  cmd_pose_subscriber_ = nh->subscribe(
      "body_pose", 100, &QuadrupedController::cmdPoseCallback_, this);

  if (publish_joint_control_) {
    joint_commands_publisher_ =
        nh->advertise<std_msgs::Float64MultiArray>(joint_control_topic, 100);
  }

  if (publish_joint_states_ && !in_gazebo_) {
    joint_states_publisher_ =
        nh->advertise<sensor_msgs::JointState>("joint_states", 100);
  }

  if (publish_foot_contacts_ && !in_gazebo_) {
    foot_contacts_publisher_ =
        nh->advertise<champ_msgs::ContactsStamped>("foot_contacts", 100);
  }

  gait_config_.knee_orientation = knee_orientation.c_str();

  base_.setGaitConfig(gait_config_);
  champ::URDF::loadFromServer(base_, nh);
  joint_names_ = champ::URDF::getJointNames(nh);

  loop_timer_ = pnh->createTimer(
      ros::Duration(loop_rate_), &QuadrupedController::controlLoopRl_, this);
  start_ = ros::WallTime::now();

  req_pose_.position.z = gait_config_.nominal_height;
}

bool QuadrupedController::selfRightRobotSrvCallback_(
    std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {
  setRobotUpright_();
  res.success = true;

  return true;
}

void QuadrupedController::setRobotUpright_() {
  if (rotMatBtoI_.row(2)[2] < 0.7) {
    pose_.model_name = gazebo_model_name_;
    pose_.pose.position.x = body_position_(0);
    pose_.pose.position.y = body_position_(1);
    pose_.pose.position.z = 0.54;

    // get the projection of the x-axis in the horizontal plane
    Eigen::Vector3d x = rotMatBtoI_.col(0);
    double theta = std::atan2(x(1), x(0));

    pose_.pose.orientation.x = 0.;
    pose_.pose.orientation.y = 0.;
    pose_.pose.orientation.z = std::sin(theta / 2.);
    pose_.pose.orientation.w = std::cos(theta / 2.);

    gazebo_msgs::SetModelState srv;
    srv.request.model_state = pose_;

    std_srvs::Empty empty_srv;

    pause_gazebo_.call(empty_srv);
    set_spot_pose_srv_.call(srv);
    unpause_gazebo_.call(empty_srv);
  }
}

void QuadrupedController::callback(
    const sensor_msgs::JointState::ConstPtr& joint_msg,
    const nav_msgs::OdometryConstPtr& ground_truth_msg) {
  jointStatesCallback_(joint_msg);

  groundTruthOdomCallback_(ground_truth_msg);

}

void QuadrupedController::controlLoopRl_(const ros::TimerEvent& event) {
  if (control_step_ % int(control_dt_ / max(loop_rate_, (start_ - ros::WallTime::now()).toSec())) == 0) {
    control_step_ = 0;

    getStateForRlLocoController_();

    getActionFromRlLocoController_();

    updateGaitParameters_();

    // publish joint angle targets
    float target_joint_positions[12];
    for (int i = 0; i < rl_controller_config_.n_joints; i++) {
      target_joint_positions[i] = (float)action_(i);
    }

    publishJoints_(target_joint_positions);
  }
  control_step_++;

  // perform self-righting, if the robot is in danger of falling over
  if (automatic_self_righting_enabled_) {
    setRobotUpright_();
  }

  start_ = ros::WallTime::now();
}

void QuadrupedController::setJointControllersGains_() {
  // TODO: Set the PID gains for the joint controllers to the same as those used
  // in the training code
}

void QuadrupedController::getStateForRlLocoController_() {
  int pos = 0;

  // base gravity axis
  rotMatBtoI_ = quatToRotMat(quat_base_to_inertial_);
  state_unscaled_.segment(pos, 3) = rotMatBtoI_.row(2);
  pos += 3;

  // joint angles
  state_unscaled_.segment(pos, rl_controller_config_.n_joints) =
      joint_position_;
  pos += rl_controller_config_.n_joints;

  // body velocities (in the body frame)
  state_unscaled_.segment(pos, 3) = body_lin_velocity_;
  pos += 3;

  state_unscaled_.segment(pos, 3) = body_ang_velocity_;
  pos += 3;

  // joint velocities
  state_unscaled_.segment(pos, rl_controller_config_.n_joints) =
      joint_velocity_;
  pos += rl_controller_config_.n_joints;

  // target velocity
  state_unscaled_(pos) = req_vel_.linear.x;
  pos++;
  state_unscaled_(pos) = req_vel_.linear.y;
  pos++;
  state_unscaled_(pos) = req_vel_.angular.z;
  pos++;

  // action history
  state_unscaled_.segment(pos, 3 * rl_controller_config_.acts_dim) =
      action_history_;
  pos += 3 * rl_controller_config_.acts_dim;

  // stride
  state_unscaled_(pos) = rl_controller_config_.gait_stride;
  pos++;

  for (int i = 0; i < 4; i++) {
    if (!is_stance_gait_) {
      // swing start
      state_unscaled_(pos) = rl_controller_config_.swing_start[i];

      // swing duration
      state_unscaled_(pos + 4) = rl_controller_config_.swing_duration[i];

      // phase time left
      state_unscaled_(pos + 8) = phase_time_left_[i];
    } else {
      // swing start
      state_unscaled_(pos) = rl_controller_config_.stance_swing_start[i];

      // swing duration
      state_unscaled_(pos + 4) = rl_controller_config_.stance_swing_duration[i];

      // phase time left
      state_unscaled_(pos + 8) = 1.;

      phase_ = 0.;
    }

    if (!is_stance_gait_) {
      // desired contact states
      state_unscaled_(pos + 12) = desired_foot_contacts_[i];

    } else {
      state_unscaled_(pos + 12) = true;

    }

    pos++;
  }
  pos += 12; // 16;

  int n_joints = rl_controller_config_.n_joints;
  int buffer_len = rl_controller_config_.buffer_len;
  if (step_ % rl_controller_config_.buffer_stride == 0) {
    step_ = 0;
    // joint position history
    Eigen::VectorXd temp;
    temp.setZero((buffer_len - 1) * n_joints);
    temp = state_unscaled_.segment(pos + n_joints, n_joints * (buffer_len - 1));
    state_unscaled_.segment(pos + n_joints * (buffer_len - 1), n_joints) =
        action_.segment(0, n_joints) - joint_position_;
    state_unscaled_.segment(pos, n_joints * (buffer_len - 1)) = temp;
    pos += n_joints * buffer_len;

    // joint velocity history
    temp.setZero((buffer_len - 1) * n_joints);
    temp = state_unscaled_.segment(pos + n_joints, n_joints * (buffer_len - 1));
    state_unscaled_.segment(pos + n_joints * (buffer_len - 1), n_joints) =
        joint_velocity_;
    state_unscaled_.segment(pos, n_joints * (buffer_len - 1)) = temp;
    pos += n_joints * buffer_len;

    temp.setZero(3 * (buffer_len - 1));
    temp = state_unscaled_.segment(pos + 3, 3 * (buffer_len - 1));
    Eigen::Vector3d bodyVelErr;
    bodyVelErr(0) = req_vel_.linear.x - body_lin_velocity_(0);
    bodyVelErr(1) = req_vel_.linear.y - body_lin_velocity_(1);
    bodyVelErr(2) = req_vel_.angular.z - body_ang_velocity_(2);
    state_unscaled_.segment(pos + 3 * (buffer_len - 1), 3) = bodyVelErr;
    state_unscaled_.segment(pos, 3 * (buffer_len - 1)) = temp;
    pos += 3 * buffer_len;
  }
  step_++;

  state_scaled_ = (state_unscaled_ - rl_controller_config_.obsMean)
                      .cwiseQuotient(rl_controller_config_.obsStd);
}

void QuadrupedController::getActionFromRlLocoController_() {
  Eigen::Matrix<double, 13, 1> action_unscaled;

  net_.forward<double, 13, 362>(action_unscaled, state_scaled_);

  action_ = action_unscaled.cwiseProduct(rl_controller_config_.actionStd) +
      rl_controller_config_.actionMean;

  Eigen::VectorXd temp;
  temp.setZero(2 * rl_controller_config_.acts_dim);
  temp = action_history_.segment(rl_controller_config_.acts_dim,
                                 2 * rl_controller_config_.acts_dim);
  action_history_.segment(0, 2 * rl_controller_config_.acts_dim) = temp;
  action_history_.segment(2 * rl_controller_config_.acts_dim,
                          rl_controller_config_.acts_dim) = action_;
}

void QuadrupedController::updateGaitParameters_() {
  if (!is_stance_gait_) {
    double freq = (1.0 / rl_controller_config_.gait_stride +
                   action_(rl_controller_config_.acts_dim - 1)) *
        rl_controller_config_.control_dt;
    phase_ = wrap_01(phase_ + freq);

    for (int i = 0; i < 4; i++) {
      double swingEnd = wrap_01(rl_controller_config_.swing_start[i] +
                                rl_controller_config_.swing_duration[i]);
      double phaseShifted = wrap_01(phase_ - swingEnd);
      double swingStartShifted = 1.0 - rl_controller_config_.swing_duration[i];

      if (phaseShifted < swingStartShifted) {
        desired_foot_contacts_[i] = true;
        phase_time_left_[i] = (swingStartShifted - phaseShifted) *
            rl_controller_config_.gait_stride;
        target_foot_clearance_[i] = 0.0;
      } else {
        desired_foot_contacts_[i] = false;
        phase_time_left_[i] =
            (1.0 - phaseShifted) * rl_controller_config_.gait_stride;
        target_foot_clearance_[i] = rl_controller_config_.max_foot_height *
            (-std::sin(2 * M_PI * phaseShifted) < 0. ?
                 0. :
                 -std::sin(2 * M_PI * phaseShifted));
      }
    }
  }
}

void QuadrupedController::groundTruthOdomCallback_(
    const nav_msgs::OdometryConstPtr& msg) {
  body_position_(0) = msg->pose.pose.position.x;
  body_position_(1) = msg->pose.pose.position.y;
  body_position_(2) = msg->pose.pose.position.z;

  quat_base_to_inertial_(0) = msg->pose.pose.orientation.w;
  quat_base_to_inertial_(1) = msg->pose.pose.orientation.x;
  quat_base_to_inertial_(2) = msg->pose.pose.orientation.y;
  quat_base_to_inertial_(3) = msg->pose.pose.orientation.z;
  quat_base_to_inertial_.normalize();

  rotMatBtoI_ = quatToRotMat(quat_base_to_inertial_);

  body_lin_velocity_(0) = msg->twist.twist.linear.x;
  body_lin_velocity_(1) = msg->twist.twist.linear.y;
  body_lin_velocity_(2) = msg->twist.twist.linear.z;
  body_lin_velocity_ = rotMatBtoI_.transpose() * body_lin_velocity_;

  body_ang_velocity_(0) = msg->twist.twist.angular.x;
  body_ang_velocity_(1) = msg->twist.twist.angular.y;
  body_ang_velocity_(2) = msg->twist.twist.angular.z;
  body_ang_velocity_ = rotMatBtoI_.transpose() * body_ang_velocity_;
}

void QuadrupedController::endEffectorContactCallback_(
    const gazebo_msgs::ContactsState::ConstPtr& msg) {
  // order of the legs is: FL, FR, RL, RR
  std::vector<std::string> foot_names = {
      "spot1::front_left_lower_leg::front_left_lower_leg_collision",
      "spot1::front_right_lower_leg::front_right_lower_leg_collision",
      "spot1::rear_left_lower_leg::rear_left_lower_leg_collision",
      "spot1::rear_right_lower_leg::rear_right_lower_leg_collision"};
  std::string ground_name = "ground_plane::link::collision";

  for (int i = 0; i < 4; i++)
    foot_contacts_[i] = false;

  for (int i = 0; i < msg->states.size(); i++) {
    std::set<std::string> contact_strings;
    contact_strings.insert(msg->states[i].collision1_name);
    contact_strings.insert(msg->states[i].collision2_name);
    for (int j = 0; j < 4; j++) {
      if (contact_strings.find(foot_names[j]) != contact_strings.end()) {
        if (contact_strings.find(ground_name) != contact_strings.end()) {
          foot_contacts_[j] = true;
        }
      }
    }
  }

}

void QuadrupedController::imuCallback_(const sensor_msgs::Imu::ConstPtr& msg) {
  quat_base_to_inertial_(0) = msg->orientation.w;
  quat_base_to_inertial_(1) = msg->orientation.x;
  quat_base_to_inertial_(2) = msg->orientation.y;
  quat_base_to_inertial_(3) = msg->orientation.z;
  quat_base_to_inertial_.normalize();

  body_ang_velocity_(0) = msg->angular_velocity.x;
  body_ang_velocity_(1) = msg->angular_velocity.y;
  body_ang_velocity_(2) = msg->angular_velocity.z;

  // transform the vectors into the correct frame according to the IMU
  // orientation given in the URDF (for now, hardcoded into the startup script).
  quat_base_to_inertial_ = quatMult(
      quat_base_to_inertial_,
      quat_base_to_imu_); // quaternion rotation from body frame to global frame
  body_ang_velocity_ = rot_base_to_imu_.transpose() *
      body_ang_velocity_; // base angular velocity expressed in base frame
}

void QuadrupedController::bodyVelocityCallback_(
    const geometry_msgs::Twist::ConstPtr& msg) {
  body_lin_velocity_(0) = msg->linear.x;
  body_lin_velocity_(1) = msg->linear.y;
  body_lin_velocity_(2) = msg->linear.z;
  body_ang_velocity_(0) = msg->angular.x;
  body_ang_velocity_(1) = msg->angular.y;
  body_ang_velocity_(2) = msg->angular.z;
}

void QuadrupedController::jointStatesCallback_(
    const sensor_msgs::JointState::ConstPtr& msg) {
  for (int i = 0; i < rl_controller_config_.n_joints; i++) {
    joint_position_(i) = msg->position[i];
    joint_velocity_(i) = msg->velocity[i];
  }
}

void QuadrupedController::controlLoop_(const ros::TimerEvent& event)
{
    float target_joint_positions[12];
    geometry::Transformation target_foot_positions[4];
    bool foot_contacts[4];

    body_controller_.poseCommand(target_foot_positions, req_pose_);
    leg_controller_.velocityCommand(target_foot_positions, req_vel_);
    kinematics_.inverse(target_joint_positions, target_foot_positions);
    
    for(size_t i = 0; i < 4; i++)
    {
        if(base_.legs[i]->gait_phase())
            foot_contacts[i] = 1;
        else
            foot_contacts[i] = 0;
    }

    publishFootContacts_(foot_contacts);
    publishJoints_(target_joint_positions);
}

void QuadrupedController::cmdVelCallback_(const geometry_msgs::Twist::ConstPtr& msg)
{
  req_vel_.linear.x = clamp(msg->linear.x,
                            -rl_controller_config_.target_vel_lim_x,
                            rl_controller_config_.target_vel_lim_x);
  req_vel_.linear.y = clamp(msg->linear.y,
                            -rl_controller_config_.target_vel_lim_y,
                            rl_controller_config_.target_vel_lim_y);
  req_vel_.angular.z = clamp(msg->angular.z,
                             -rl_controller_config_.target_vel_lim_r,
                             rl_controller_config_.target_vel_lim_r);

  // if ((req_vel_.linear.x * req_vel_.linear.x +
  //      req_vel_.linear.y * req_vel_.linear.y +
  //      req_vel_.angular.z * req_vel_.angular.z) < 0.1) {
  //   is_stance_gait_ = true;

  //   req_vel_.linear.x = 0.;
  //   req_vel_.linear.y = 0.;
  //   req_vel_.angular.z = 0.;
  // } else {
  //   is_stance_gait_ = false;
  // }
}

void QuadrupedController::cmdPoseCallback_(const geometry_msgs::Pose::ConstPtr& msg)
{
  tf::Quaternion base_quat;
  base_quat[0] = msg->orientation.x;
  base_quat[1] = msg->orientation.y;
  base_quat[2] = msg->orientation.z;
  base_quat[3] = msg->orientation.w;

  tf::Matrix3x3 m(base_quat);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  req_pose_.orientation.roll = roll;
  req_pose_.orientation.pitch = pitch;
  req_pose_.orientation.yaw = yaw;

  req_pose_.position.x = msg->position.x;
  req_pose_.position.y = msg->position.y;
  req_pose_.position.z = msg->position.z + gait_config_.nominal_height;
}

void QuadrupedController::publishJoints_(float target_joints[12])
{
    if(publish_joint_control_)
    {
      std_msgs::Float64MultiArray joints_cmd_msg;

      joints_cmd_msg.data.resize(12);
      for (size_t i = 0; i < 12; i++) {
        joints_cmd_msg.data[i] = target_joints[i];
        }

        joint_commands_publisher_.publish(joints_cmd_msg);
    }

    if(publish_joint_states_ && !in_gazebo_)
    {
        sensor_msgs::JointState joints_msg;

        joints_msg.header.stamp = ros::Time::now();
        joints_msg.name.resize(joint_names_.size());
        joints_msg.position.resize(joint_names_.size());
        joints_msg.name = joint_names_;

        for (size_t i = 0; i < joint_names_.size(); ++i)
        {    
            joints_msg.position[i]= target_joints[i];
        }

        joint_states_publisher_.publish(joints_msg);
    }
}

void QuadrupedController::publishFootContacts_(bool foot_contacts[4])
{
    if(publish_foot_contacts_ && !in_gazebo_)
    {
        champ_msgs::ContactsStamped contacts_msg;
        contacts_msg.header.stamp = ros::Time::now();
        contacts_msg.contacts.resize(4);

        for(size_t i = 0; i < 4; i++)
        {
            contacts_msg.contacts[i] = foot_contacts[i];
        }

        foot_contacts_publisher_.publish(contacts_msg);
    }
}
