//
// Created by jemin on 3/27/19.
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

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "yaml-cpp/yaml.h"

namespace raisim {

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir) {
    cfg_ = YAML::Load(cfg);
    if (cfg_["render"].template as<bool>())
      render_ = cfg_["render"].template as<bool>();

    raisim::World::setActivationKey(cfg_["raisim_activation_key"].template as<std::string>());

  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  void init() {
    omp_set_num_threads(cfg_["num_threads"].template as<int>());
    num_envs_ = cfg_["num_envs"].template as<int>();
    setSeed(cfg_["seed"].template as<int>());

    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template as<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template as<double>());
    }

    if (render_) raisim::OgreVis::get()->hideWindow();

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    // extraInfoDim_ = environments_[0]->getExtraInfoDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0,
               "Observation/Action dimension must be defined in the constructor of each environment!")

    /// generate reward names
    /// compute it once to get reward names. actual value is not used
    // environments_[0]->updateExtraInfo();
    // for (auto &re: environments_[0]->extraInfo_)
    //   extraInfoName_.push_back(re.first);
  }


  // resets all environments and returns observation
  void reset(Eigen::Ref<EigenRowMajorMat> &ob) {
    for (auto env: environments_)
      env->reset();

    observe(ob);
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_envs_; i++) {
      environments_[i]->observe(ob.row(i));
    }
  }

  void setGaitString(std::vector<std::string> gaitStr){
    for (int i = 0; i < num_envs_; i++) {
      environments_[i]->setGait(gaitStr[i]);
    }

    return;
  }

  pybind11::array getGaitString() {
    std::vector<std::string> gait;
    std::string gaitStr;

// #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_envs_; i++) {
      environments_[i]->getGaitString(gaitStr);
      gait.push_back(gaitStr);
    }

    return pybind11::array(pybind11::cast(gait));
  }

  void getExtraInfo(Eigen::Ref<EigenRowMajorMat>& extraInfo) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_envs_; i++) {
      environments_[i]->getExtraInfo(extraInfo.row(i));
    }
  }

//   void setTargetVelocity(Eigen::Ref<EigenRowMajorMat>& targetVelocity) {
// #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < num_envs_; i++) {
//       environments_[i]->setTargetVelocityExternally(targetVelocity.row(i));
//     }
//   }

//   void observeStudent(Eigen::Ref<EigenRowMajorMat> &ob) {
// #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < num_envs_; i++) {
//       environments_[i]->observeStudent(ob.row(i));
//     }
//   }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenRowMajorMat> &ob,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_envs_; i++) {
      perAgentStep(i, action, ob, reward, done);
      environments_[i]->observe(ob.row(i));
    }
  }

  void testStep(Eigen::Ref<EigenRowMajorMat> &action,
                Eigen::Ref<EigenRowMajorMat> &ob,
                Eigen::Ref<EigenVec> &reward,
                Eigen::Ref<EigenBoolVec> &done) {
    if (render_) environments_[0]->turnOnVisualization();
    perAgentStep(0, action, ob, reward, done);
    if (render_) environments_[0]->turnOffvisualization();

    environments_[0]->observe(ob.row(0));
  }

  void startRecordingVideo(const std::string &fileName) {
    if (render_) environments_[0]->startRecordingVideo(fileName);
  }

  void stopRecordingVideo() {
    if (render_) environments_[0]->stopRecordingVideo();
  }

  void showWindow() {
    raisim::OgreVis::get()->showWindow();
  }

  void hideWindow() {
    raisim::OgreVis::get()->hideWindow();
  }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec> &terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  // int getStudentObDim() { return environments_[0]->getStudentObDim(); }
  // int getTimeSeriesLen() { return environments_[0]->getTimeSeriesLen(); }
  // int getPartialObsDim() { return environments_[0]->getPartialObsDim(); }
  // int getExtrasDim() { return environments_[0]->getExtrasDim(); }
  int getActionDim() { return actionDim_; }
  // int getExtraInfoDim() { return extraInfoDim_; }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

 private:

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenRowMajorMat> &ob,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId));

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    // for (int j = 0; j < extraInfoName_.size(); j++)
    //   extraInfo(agentId, j) = environments_[agentId]->extraInfo_[extraInfoName_[j]];

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }

  std::vector<ChildEnvironment *> environments_;
  std::vector<std::string> extraInfoName_;

  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0, extraInfoDim_ = 0;
  bool recordVideo_ = false, render_ = false;
  std::string resourceDir_;
  YAML::Node cfg_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
