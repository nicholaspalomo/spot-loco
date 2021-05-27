//
// Created by jemin on 11/9/19.
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef _RAISIM_GYM_ANYMAL_JOINTSPEEDTORQUELOGGER_HPP
#define _RAISIM_GYM_ANYMAL_JOINTSPEEDTORQUELOGGER_HPP
#include "raisim/imgui_plot.h"
#include "SlidingMemory.hpp"

namespace raisim {

  class JointSpeedTorqueLogger {

  public:
    JointSpeedTorqueLogger(){
      this->init();
    };

    static const int numberOfJoints_ = 16;
    // std::vector<SlidingMemory> jointSpeed_, jointTorque_, targetHeightReached_, bodyVel_, footTargetFunc_, footHeight_;
    std::vector<SlidingMemory> footTargetFunc_, footHeight_;
    // const float *speed_data_[numberOfJoints_], *torque_data_[numberOfJoints_], *body_vel_data_[6], *target_height_reached_data_[1], *foot_target_func_data_[4], *foot_height_data_[4];
    const float *foot_target_func_data_[4], *foot_height_data_[4];
    std::unique_ptr<SlidingMemory> time_;

    ImFont* fontBig;
    ImFont* fontMid;
    ImFont* fontSmall;

    void init(int buffer_size = 100) {
      // jointSpeed_.resize(numberOfJoints_, SlidingMemory(buffer_size, 0.));
      // jointTorque_.resize(numberOfJoints_, SlidingMemory(buffer_size, 0.));
      // targetHeightReached_.resize(1, SlidingMemory(buffer_size, 0.));
      // bodyVel_.resize(6, SlidingMemory(buffer_size, 0.));
      footTargetFunc_.resize(4, SlidingMemory(buffer_size, 0.));
      footHeight_.resize(4, SlidingMemory(buffer_size, 0.));
      time_ = std::make_unique<SlidingMemory>(buffer_size, 0.);
    }

    void push_back(double time_point,
                  // const Eigen::VectorXd &joint_speed,
                  // const Eigen::VectorXd &joint_torque,
                  // const Eigen::VectorXd &target_vel,
                  // const Eigen::VectorXd &current_body_vel,
                  // const float &target_height_reached,
                  const double (&foot_target_func)[4],
                  const double (&foot_height)[4])
    {
      time_->push_back(float(time_point));

      // for (int i = 0; i < numberOfJoints_; i++) {
      //   jointSpeed_[i].push_back(float(joint_speed[i]));
      //   jointTorque_[i].push_back(float(joint_torque[i]));
      // }

      // for (int i = 0; i < 3; i++) {
      //   bodyVel_[i].push_back(float(target_vel[i]));
      // }
      // for (int i = 3; i < 6; i++) {
      //   bodyVel_[i].push_back(float(current_body_vel[i]));
      // }

      // targetHeightReached_[0].push_back(target_height_reached);

      for (int i = 0; i < 4; i++) {
        footTargetFunc_[i].push_back(float(foot_target_func[i]));
        footHeight_[i].push_back(float(foot_height[i]));
      }
    }

    void clean() {
      time_->clear();

      // for (int i = 0; i < numberOfJoints_; i++) {
      //   jointSpeed_[i].clear();
      //   jointTorque_[i].clear();
      // }

      // for (int i = 0; i < 6; i++) {
      //   bodyVel_[i].clear();
      // }

      // targetHeightReached_[0].clear();

      for (int i = 0; i < 4; i++) {
        footTargetFunc_[i].clear();
        footHeight_[i].clear();
      }
    }

    void callback() {
      // ADD MORE COLORS FOR DIFFERENT A NUMBER OF LEGS //
      static ImU32 colors[numberOfJoints_] = {ImColor(114, 229, 239),
                                            ImColor(52, 115, 131),
                                            ImColor(111, 239, 112),
                                            ImColor(30, 123, 32),
                                            ImColor(201, 221, 135),
                                            ImColor(137, 151, 91),
                                            ImColor(233, 173, 111),
                                            ImColor(159, 88, 39),
                                            ImColor(214, 68, 5),
                                            ImColor(235, 62, 134),
                                            ImColor(142, 0, 73),
                                            ImColor(191, 214, 250),
                                            ImColor(171, 228, 250),
                                            ImColor(165, 214, 272),
                                            ImColor(181, 229, 256),
                                            ImColor(195, 222, 250)};

      static uint32_t selection_start = 0, selection_length = 0;

      // for (int i = 0; i < numberOfJoints_; i++) {
      //   speed_data_[i] = jointSpeed_[i].data();
      //   torque_data_[i] = jointTorque_[i].data();
      // }

      // for (int i = 0; i < 6; i++) {
      //   body_vel_data_[i] = bodyVel_[i].data();
      // }

      // target_height_reached_data_[0] = targetHeightReached_[0].data();

      for (int i = 0; i < 4; i++) {
        foot_target_func_data_[i] = footTargetFunc_[i].data();
        foot_height_data_[i] = footHeight_[i].data();
      }

      // Draw first plot with multiple sources
      ImGui::PlotConfig conf;
      conf.values.xs = time_->data();
      conf.values.count = 100; // buffer size
      conf.values.ys_count = 4;
      conf.values.ys_list = &foot_target_func_data_[0];
      conf.values.colors = colors;
      conf.scale.min = -0.2;
      conf.scale.max = 0.2;
      conf.grid_x.show = false;
      conf.grid_x.size = 5;
      conf.grid_x.subticks = 2;
      conf.selection.show = true;
      conf.selection.start = &selection_start;
      conf.selection.length = &selection_length;
      conf.grid_y.show = false;
      conf.grid_y.subticks = 2;
      conf.selection.show = true;
      conf.frame_size = ImVec2(400, 100);
      conf.grid_y.size = 1.0f;
      // ImGui::PushFont(fontSmall);
      // ImGui::Text("Joint position");
      // ImGui::PopFont();
      ImGui::Plot("plot1", conf);

      conf.values.xs = time_->data();
      conf.values.count = 100; // buffer size
      conf.values.ys_count = 4;
      conf.values.ys_list = &foot_height_data_[0];
      conf.values.colors = colors;
      conf.scale.min = -0.2;
      conf.scale.max = 0.2;
      conf.grid_x.show = false;
      conf.grid_x.size = 5;
      conf.grid_x.subticks = 2;
      conf.selection.show = true;
      conf.selection.start = &selection_start;
      conf.selection.length = &selection_length;
      conf.grid_y.show = false;
      conf.grid_y.subticks = 2;
      conf.selection.show = true;
      conf.frame_size = ImVec2(400, 100);
      conf.grid_y.size = 1.0f;
      // ImGui::PushFont(fontSmall);
      // ImGui::Text("Joint position");
      // ImGui::PopFont();
      ImGui::Plot("plot2", conf);




      // ImGui::Plot("plot1", conf);

      // Draw second plot with the selection
      // reset previous values
      // conf.values.ys_list = &torque_data_[0];
      // conf.selection.show = false;
      // conf.scale.min = -2.;
      // conf.scale.max = 2.;
      // conf.grid_y.size = 20.0f;
      // ImGui::PushFont(fontSmall);
      // ImGui::Text("Joint position");
      // ImGui::PopFont();
      // ImGui::Plot("plot2", conf);

      // Draw third plot with the selection
      // reset previous values
      // conf.values.count = targetHeightReached_[0].size();
      // conf.values.ys_count = 1;
      // conf.values.ys_list = &target_height_reached_data_[0];
      // conf.selection.show = false;
      // conf.scale.min = 0.;
      // conf.scale.max = 2.;
      // conf.grid_y.size = 20.0f;
      // ImGui::PushFont(fontSmall);
      // ImGui::Text("Joint position");
      // ImGui::PopFont();
      // ImGui::Plot("plot3", conf);

      // Draw fourth plot with the selection
      // reset previous values
      // conf.values.ys_count = 6;
      // conf.values.ys_list = &body_vel_data_[0];
      // conf.selection.show = false;
      // conf.scale.min = -1.;
      // conf.scale.max = 1.;
      // conf.grid_y.size = 1.0f;
      // ImGui::PushFont(fontSmall);
      // ImGui::Text("Joint position");
      // ImGui::PopFont();
      // ImGui::Plot("plot4", conf);

      // Foot height
      // conf.values.ys_count = 4;
      // conf.values.ys_list = &foot_height_data_[0];
      // conf.selection.show = false;
      // conf.scale.min = 0.;
      // conf.scale.max = .3;
      // conf.grid_y.size = 1.0f;
      // ImGui::PushFont(fontSmall);
      // ImGui::Text("Joint position");
      // ImGui::PopFont();
      // ImGui::Plot("plot6", conf);

    }
  };
}

#endif //_RAISIM_GYM_ANYMAL_JOINTSPEEDTORQUELOGGER_HPP