// Copyright (c) 2021, Nico Palomo
//
/* Convention
*
*   observation space = [ height                                                      n =  1, si =  0
*                         z-axis in world frame expressed in body frame (R_b.row(2))  n =  3, si =  1
*                         joint angles,                                               n = 12, si =  4
*                         body Linear velocities,                                     n =  3, si = 16
*                         body Angular velocities,                                    n =  3, si = 19
*                         joint velocities,                                           n = 12, si = 22 ] total 34
*
*/


#include <stdlib.h>
#include <cstdint>
#include <set>
#include <random> // for random number generator
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnv.hpp"
#include "visSetupCallback.hpp"

#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/helper.hpp"
#include "visualizer/guiState.hpp"
#include "visualizer/raisimBasicImguiPanel.hpp"


namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const YAML::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), 
      distribution_(0.0, 0.2), 
      visualizable_(visualizable), 
      uniformDist_(0.0, 1.0),
      randNumGen_(std::chrono::system_clock::now().time_since_epoch().count()) {

    /// add objects
    spot_ = world_->addArticulatedSystem(resourceDir_+"/spot_description/spot.urdf");
    // spot_->printOutBodyNamesInOrder(); // for debug
    // spot_->printOutFrameNamesInOrder(); // for debug
    spot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    auto ground = world_->addGround();
    world_->setERP(0.2,0);

    /// generate a height map
    heightMap_ = generateTerrain(cfg);

    /// get robot data
    gcDim_ = spot_->getGeneralizedCoordinateDim();
    gvDim_ = spot_->getDOF();
    nJoints_ = 12;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    torque_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(actionDim_);
    targetVelocity_.setZero();

    /// this is nominal configuration of spot (FL, FR, RL, RR)
    gc_init_ << 0, 0, heightMap_->getHeight(0., 0.) + 0.2 + 0.54, 1.0, 0.0, 0.0, 0.0, -0.1, 1.10, -1.90, 0.1, 1.10, -1.90, -0.1, 1.10, -1.90, 0.1, 1.10, -1.90;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(40.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(1.0);
    spot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    READ_YAML(int, bufferLength_, cfg["bufferLength"])
    READ_YAML(int, bufferStride_, cfg["bufferStride"])

    actionDim_ = nJoints_ + 1;
    obDim_ = 1 + 3 + 12 + 3 + 3 + 12 + 3 + 3*actionDim_ + 1 + 4*5 + bufferLength_ * (12 + 12) - 4 + 3 * bufferLength_ - 1; /// convention described on top
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);
    actionHistory_.setZero(3*actionDim_); actionHistory_ << gc_init_.tail(nJoints_), 0, gc_init_.tail(nJoints_), 0, gc_init_.tail(nJoints_), 0; obDouble_.setZero(obDim_); obScaled_.setZero(obDim_); obNoise_.setZero(obDim_);

    /// Command (body velocity targets)
    getRangeFromYaml<double>(commandParams_["x"], cfg["target"]["x"]);
    getRangeFromYaml<double>(commandParams_["y"], cfg["target"]["y"]);
    getRangeFromYaml<double>(commandParams_["yaw"], cfg["target"]["yaw"]);

    /// action & observation scaling
    actionMean_ << gc_init_.tail(nJoints_), 0.;
    actionStd_.setConstant(0.6);
    actionStd_(nJoints_) = 0.1;

    obMean_ << 
        // 0.48, /// average height
        0.0, 0.0, 0.0, /// gravity axis 3
        gc_init_.tail(12), /// joint position 12
        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
        Eigen::VectorXd::Constant(12, 0.0), /// joint vel history
        Eigen::VectorXd::Constant(3, 0.0), /// target velocity
        Eigen::VectorXd::Constant(3*actionDim_, 0.0), /// action history
        0.0, /// stride
        Eigen::VectorXd::Constant(4, 0.0), /// swing start
        Eigen::VectorXd::Constant(4, 0.0), /// swing duration
        Eigen::VectorXd::Constant(4, 0.0), /// phase time left
        Eigen::VectorXd::Constant(4, 0.0), /// desired contact states
        // Eigen::VectorXd::Constant(4, 0.0), /// target foot clearance
        Eigen::VectorXd::Constant(nJoints_ * bufferLength_, 0.0), /// joint position history
        Eigen::VectorXd::Constant(nJoints_ * bufferLength_, 0.0), /// joint velocity history
        Eigen::VectorXd::Constant(3 * bufferLength_, 0.); // body velocity error

    obNoise_ << 
      // 0.02,
      Eigen::VectorXd::Constant(3, 0.02),
      Eigen::VectorXd::Constant(12, 0.02),
      Eigen::VectorXd::Constant(6, 0.02),
      Eigen::VectorXd::Constant(12, 0.02),
      Eigen::VectorXd::Constant(3, 0.0),
      Eigen::VectorXd::Constant(3*actionDim_, 0.0),
      0.,
      Eigen::VectorXd::Constant(4, 0.0), /// swing start
      Eigen::VectorXd::Constant(4, 0.0), /// swing duration
      Eigen::VectorXd::Constant(4, 0.0), /// phase time left
      Eigen::VectorXd::Constant(4, 0.0), /// desired contact states
      // Eigen::VectorXd::Constant(4, 0.0), /// target foot clearance
      Eigen::VectorXd::Constant(nJoints_ * bufferLength_, 0.0), /// joint position history
      Eigen::VectorXd::Constant(nJoints_ * bufferLength_, 0.0), /// joint velocity history
      Eigen::VectorXd::Constant(3 * bufferLength_, 0.); // body velocity error

    obStd_ << 
        // 0.12, /// average height
        Eigen::VectorXd::Constant(3, 0.7), /// gravity axes angles
        Eigen::VectorXd::Constant(12, 1.0 / 1.0), /// joint angles
        Eigen::VectorXd::Constant(3, 2.0), /// linear velocity
        Eigen::VectorXd::Constant(3, 4.0), /// angular velocities
        Eigen::VectorXd::Constant(12, 10.0), /// joint velocities
        commandParams_["x"][1], commandParams_["y"][1], commandParams_["yaw"][1],
        Eigen::VectorXd::Constant(3*actionDim_, 1.0),
        1.0, /// stride
        Eigen::VectorXd::Constant(4, 1.0), /// swing start
        Eigen::VectorXd::Constant(4, 1.0), /// swing duration
        Eigen::VectorXd::Constant(4, 1.0), /// phase time left
        Eigen::VectorXd::Constant(4, 1.0), /// desired contact states
        // Eigen::VectorXd::Constant(4, 0.2), /// target foot clearance
        Eigen::VectorXd::Constant(nJoints_ * bufferLength_, 1.0), /// joint position history
        Eigen::VectorXd::Constant(nJoints_ * bufferLength_, 10.0), /// joint velocity history
        Eigen::VectorXd::Constant(3 * bufferLength_, 1.); // body velocity error

    stanceGaitParams_.isStanceGait = true;
    for(int i = 0; i < 4; i++){
      stanceGaitParams_.swingStart[i] = 0.;
      stanceGaitParams_.swingDuration[i] = 0.;
      stanceGaitParams_.footTarget[i] = 0.;
    }

    /// Reward coefficients
    READ_YAML(double, torqueRewardCoeff_, cfg["torqueRewardCoeff"])
    READ_YAML(double, targetFollowingRewardCoeff_, cfg["targetFollowingRewardCoeff"])
    READ_YAML(double, actionSmoothnessRewardCoeff_, cfg["actionSmoothnessRewardCoeff"])
    READ_YAML(double, bodyOrientationRewardCoeff_, cfg["bodyOrientationRewardCoeff"])
    READ_YAML(double, symmetryRewardCoeff_, cfg["symmetryRewardCoeff"])
    READ_YAML(double, footClearanceRewardCoeff_, cfg["footClearanceRewardCoeff"])
    READ_YAML(double, gaitFollowingRewardCoeff_, cfg["gaitFollowingRewardCoeff"])
    READ_YAML(double, verticalVelocityRewardCoeff_, cfg["verticalVelocityRewardCoeff"])
    READ_YAML(double, bodyRatesRewardCoeff_, cfg["bodyRatesRewardCoeff"])
    READ_YAML(double, bodyHeightRewardCoeff_, cfg["bodyHeightRewardCoeff"])
    READ_YAML(double, bodyHeightTarget_, cfg["bodyHeightTarget"])
    READ_YAML(double, internalContactRewardCoeff_, cfg["internalContactRewardCoeff"])
    READ_YAML(double, jointVelocityRewardCoeff_, cfg["jointVelocityRewardCoeff"])

    gui::rewardLogger.init({"torqueReward", "targetFollowingReward", "actionSmoothnessReward", "bodyOrientationReward", "symmetryReward", "footClearanceReward", "gaitFollowingReward", "verticalVelocityReward", "bodyRatesReward", "bodyHeightReward", "internalContactReward", "jointVelocityReward"});

    /// indices of links that should not make contact with ground
    footIndices_.insert(spot_->getBodyIdx("front_left_lower_leg"));
    footIndices_.insert(spot_->getBodyIdx("front_right_lower_leg"));
    footIndices_.insert(spot_->getBodyIdx("rear_left_lower_leg"));
    footIndices_.insert(spot_->getBodyIdx("rear_right_lower_leg"));

    /// visualize if it is the first environment
    if (visualizable_) {
      auto vis = raisim::OgreVis::get();

      /// these method must be called before initApp
      vis->setWorld(world_.get());
      vis->setWindowSize(1280, 720);
      vis->setImguiSetupCallback(imguiSetupCallback);
      vis->setImguiRenderCallback(imguiRenderCallBack);
      vis->setKeyboardCallback(raisimKeyboardCallback);
      vis->setSetUpCallback(setupCallback);
      vis->setAntiAliasing(2);

      /// starts visualizer thread
      vis->initApp();

      spotVisual_ = vis->createGraphicalObject(spot_, "Spot");

      vis->createGraphicalObject(heightMap_, "heightMap");
      // vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
      desired_fps_ = 60.;
      vis->setDesiredFPS(desired_fps_);

      vis->select(spotVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void reset() final {
    genForce_.setZero(gvDim_);
    pGain_.setZero(nJoints_); pGain_.setConstant(100.);
    dGain_.setZero(nJoints_); dGain_.setConstant(4.);
    iGain_.setZero(nJoints_); iGain_.setConstant(0.);
    errCurr_.setZero(nJoints_); errPrev_.setZero(nJoints_);
    errInt_.setZero(nJoints_);

    spot_->setState(gc_init_, gv_init_);
    sampleVelocity();
    updateGaitParameters();
    updateObservation();
    if(visualizable_)
      gui::rewardLogger.clean();
      raisim::gui::jointSpeedTorqueLogger.clean();
  }

  void sampleVelocityTarget(){
    if(gaitParams_.isStanceGait){
      if(sampleUniform(0., 1.) > 0.98){
        sampleVelocity();
      }
    }else{
      if(sampleUniform(0., 1.) > 0.998){
        sampleVelocity();
      }
    }

    if(sampleUniform(0., 1.) > 0.998){
      if(!gaitParams_.isStanceGait){
        gaitParams_.isStanceGait = true;
        targetVelocity_ << 0., 0., 0.;
        defaultGaitParams_ = gaitParams_;
        gaitParams_ = stanceGaitParams_;
      }
    }
  }

  void sampleVelocity(){
    defaultGaitParams_.isStanceGait = false;
    gaitParams_ = defaultGaitParams_;

    targetVelocity_[0] = sampleUniform(commandParams_["x"][0], commandParams_["x"][1]);
    targetVelocity_[1] = sampleUniform(commandParams_["y"][0], commandParams_["y"][1]);
    targetVelocity_[2] = sampleUniform(commandParams_["yaw"][0], commandParams_["yaw"][1]);

    if(uniformDist_(randNumGen_) < 0.1){
      targetVelocity_[0] = 0.0;
    }
    if(uniformDist_(randNumGen_) < 0.1){
      targetVelocity_[1] = 0.0;
    }
    if(uniformDist_(randNumGen_) < 0.1){
      targetVelocity_[2] = 0.0;
    }
  }

  void updateGaitParameters() {

    if(!gaitParams_.isStanceGait){
      /// update phase
      double freq = (1.0 / gaitParams_.stride + gaitFreq_) * control_dt_;
      gaitParams_.phase = wrap_01(gaitParams_.phase + freq);

      // Get the current gait parameters
      for (int i = 0; i < 4; i++) {
        // remap for convenience: e.g., [.........:::::::.....] -> [..............:::::::]
        double swingEnd = wrap_01(gaitParams_.swingStart[i] + gaitParams_.swingDuration[i]);
        double phaseShifted = wrap_01(gaitParams_.phase - swingEnd);
        double swingStartShifted = 1.0 - gaitParams_.swingDuration[i];

        if (phaseShifted < swingStartShifted) { // stance phase
          gaitParams_.desiredContactStates[i] = 1.0;
          gaitParams_.phaseTimeLeft[i] = (swingStartShifted - phaseShifted) * gaitParams_.stride;
          gaitParams_.footTarget[i] = 0.0;
        } else {
          gaitParams_.desiredContactStates[i] = 0.0;
          gaitParams_.phaseTimeLeft[i] = (1.0 - phaseShifted) * gaitParams_.stride;
          gaitParams_.footTarget[i] = gaitParams_.maxFootHeight * ( -std::sin(2 * M_PI * phaseShifted) < 0. ? 0. : -std::sin(2 * M_PI * phaseShifted) );
        }
      } 
    }

  }

  void setSimulationParam(const Eigen::Ref<EigenVec>& params, std::vector<std::string> params_string) {
    
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_.head(nJoints_);
    gaitFreq_ = pTarget12_(nJoints_);

    Eigen::VectorXd temp; temp.setZero(2*actionDim_);
    temp = actionHistory_.tail(2*actionDim_);
    actionHistory_.tail(actionDim_) = pTarget12_;
    actionHistory_.head(2*actionDim_) = temp;

    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);

    for(int i=0; i<loopCount; i++) {
      spot_->getState(gc_, gv_);

      errCurr_ = pTarget_.tail(nJoints_) - gc_.tail(nJoints_);

      genForce_.tail(nJoints_) = pGain_.cwiseProduct(errCurr_) + iGain_.cwiseProduct(errInt_) + dGain_.cwiseProduct(errCurr_ - errPrev_) / simulation_dt_;

      spot_->setGeneralizedForce(genForce_);

      errPrev_ = errCurr_;
      errInt_ += simulation_dt_ * errCurr_;

      world_->integrate();

      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
        raisim::OgreVis::get()->renderOneFrame();

      visualizationCounter_++;
    }

    updateObservation();

    /// reward for minimizing torque
    torqueReward_ = torqueRewardCoeff_ * spot_->getGeneralizedForce().squaredNorm();

    /// reward for following target velocity
    if(targetVelocity_.norm() < 1e-5){
      float xVelError = (bodyLinearVel_ - targetVelocity_)[0] / commandParams_["x"][1];
      float yVelError = (bodyLinearVel_ - targetVelocity_)[1] / commandParams_["y"][1];
      float yawRateError = (bodyAngularVel_ - targetVelocity_)[2] / commandParams_["yaw"][1];

      targetFollowingReward_ = std::exp(-1.5 * (xVelError * xVelError +  yVelError * yVelError +  yawRateError * yawRateError));
    } else {
      Eigen::Vector3d currVelVecNorm = {bodyLinearVel_[0]/commandParams_["x"][1], bodyLinearVel_[1]/commandParams_["y"][1], bodyAngularVel_[2]/commandParams_["yaw"][1]};
      Eigen::Vector3d targetVelVecNorm = {targetVelocity_[0]/commandParams_["x"][1], targetVelocity_[1]/commandParams_["y"][1], targetVelocity_[2]/commandParams_["yaw"][1]};

      float dotP = currVelVecNorm[0]*targetVelVecNorm[0] + currVelVecNorm[1]*targetVelVecNorm[1] + currVelVecNorm[2]*targetVelVecNorm[2];
      dotP /= targetVelVecNorm.squaredNorm();

      targetFollowingReward_ = std::exp(-2. * (1 - dotP) * (1 - dotP));
    }
    targetFollowingReward_ *= targetFollowingRewardCoeff_;

    /// reward for action smoothness
    double secondDer = (actionHistory_.segment(2*actionDim_, actionDim_-1) - 2 * actionHistory_.segment(actionDim_, actionDim_-1) + actionHistory_.head(actionDim_-1)).squaredNorm();
    double firstDer = (actionHistory_.segment(2*actionDim_, actionDim_-1) - actionHistory_.segment(actionDim_, actionDim_-1)).squaredNorm();
    actionSmoothnessReward_ = actionSmoothnessRewardCoeff_ * (secondDer + firstDer);

    /// body orientation
    bodyOrientationReward_ = bodyOrientationRewardCoeff_ * std::exp(-700 * (1 - Rb_.row(2)[2]) * (1 - Rb_.row(2)[2]));

    /// symmetry positions
    symmetryReward_ = symmetryRewardCoeff_ * ((pTarget12_.head(nJoints_).segment(1,2) - pTarget12_.head(nJoints_).segment(10,2)).squaredNorm() + (pTarget12_.head(nJoints_).segment(4,2) - pTarget12_.head(nJoints_).segment(7,2)).squaredNorm()); // + (pTarget12_.head(nJoints_) - gc_init_.tail(nJoints_)).squaredNorm());
    // symmetryReward_ = symmetryRewardCoeff_ * (pTarget12_.head(nJoints_) - gc_init_.tail(nJoints_)).squaredNorm();


    footClearanceReward_ = 0.;
    gaitFollowingReward_ = 0.;
    for(int i = 0; i < 4; i++){
      /// foot clearance
      if(!gaitParams_.desiredContactStates[i])
        footClearanceReward_ += footClearanceRewardCoeff_ * std::exp(-100. * (gaitParams_.footTarget[i] - gaitParams_.footPosition[i]) * (gaitParams_.footTarget[i] - gaitParams_.footPosition[i]));

        if(gaitParams_.footPosition[i] > gaitParams_.footTarget[i])
          footClearanceReward_ -= (gaitParams_.footTarget[i] - gaitParams_.footPosition[i]) * (gaitParams_.footTarget[i] - gaitParams_.footPosition[i]);

      /// gait following
      if(gaitParams_.footContactStates[i] == gaitParams_.desiredContactStates[i])
        gaitFollowingReward_ += gaitFollowingRewardCoeff_;
      else
        gaitFollowingReward_ -= gaitFollowingRewardCoeff_;
    }

    /// vertical velocity
    verticalVelocityReward_ = verticalVelocityRewardCoeff_ * std::exp(-100 * gv_[2] * gv_[2]);

    /// body rates
    bodyRatesReward_ = bodyRatesRewardCoeff_ * std::exp(-0.5 * gv_.segment(3, 2).squaredNorm());

    /// body height
    bodyHeightReward_ = bodyHeightRewardCoeff_ * std::exp(-100 * (bodyHeight_ - bodyHeightTarget_) * (bodyHeight_ - bodyHeightTarget_));

    /// internal contacts
    internalContactReward_ = 0.;
    for(auto &contact : spot_->getContacts()) {
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end() || contact.isSelfCollision()) {
        internalContactReward_ += internalContactRewardCoeff_; // if we have an internal contact
      }
    }

    jointVelocityReward_ = jointVelocityRewardCoeff_ * gv_.tail(nJoints_).squaredNorm();

    updateVisualization();

    updateGaitParameters();
    sampleVelocityTarget();

    return torqueReward_ + targetFollowingReward_ + actionSmoothnessReward_ + bodyOrientationReward_ + symmetryReward_ + footClearanceReward_ + gaitFollowingReward_ + verticalVelocityReward_ + bodyRatesReward_ + bodyHeightReward_ + internalContactReward_ + jointVelocityReward_;
  }

  void updateVisualization() {
    if(visualizeThisStep_) {
      /// Update the contact state logger in the GUI
      raisim::gui::gaitLogger.appendContactStates(gaitParams_.footContactStates, gaitParams_.desiredContactStates);

      /// Update target information in GUI
      std::ostringstream out;
      out << std::fixed << std::setprecision(2);
      out << "Target [" << targetVelocity_[0] << "," << targetVelocity_[1] << "," << targetVelocity_[2] << "]\n";
      out << "Curr. Vel. [" << bodyLinearVel_[0] << "," << bodyLinearVel_[1] << "," << bodyAngularVel_[2] << "]\n";
      out << "Gait Freq. [" << gaitFreq_ << "]\n";
      raisim::gui::VisualizerString2 = out.str();

      // update the command arrow visual display in the GUI
      auto &list = raisim::OgreVis::get()->getVisualObjectList();
      Eigen::Vector3d linVelTarget;
      Eigen::Vector3d offsetVec;
      raisim::Vec<3> targetVec;
      raisim::Vec<3> currentLinVelVec;
      raisim::Mat<3, 3> rotMat;
      offsetVec << gc_[0], gc_[1], gc_[2]
          + 0.5; ///< offset the target vector 1/2 meter above the center of mass

      {
        auto &arrow = list["command_arrow1"];
        auto &arrowCurrentLinVel = list["current_arrow1"];
        {
          targetVec.e() << targetVelocity_[0], targetVelocity_[1], 0.;
          targetVec.e().normalize();
          targetVec.e() = Rb_.e() * targetVec.e();
          raisim::zaxisToRotMat(targetVec, arrow.rotationOffset);
          arrow.offset.e() = offsetVec;
          double scale = targetVelocity_.head(2).norm();
          arrow.scale.e().setConstant(scale * 0.2);

          currentLinVelVec.e() << gv_[0], gv_[1], 0.;
          scale = currentLinVelVec.e().head(2).norm();
          currentLinVelVec.e().normalize();
          raisim::zaxisToRotMat(currentLinVelVec, arrowCurrentLinVel.rotationOffset);
          arrowCurrentLinVel.offset.e() = offsetVec;
          arrowCurrentLinVel.scale.e().setConstant(scale * 0.2);

        }
      }

      {
        auto &arrow = list["command_arrow2"];
        auto &arrowCurrentYawRate = list["current_arrow2"];
        linVelTarget[2] = targetVelocity_[2];
        {
          targetVec.e() << 0., 0., linVelTarget[2];
          targetVec.e().normalize();
          targetVec.e() = Rb_.e() * targetVec.e();
          raisim::zaxisToRotMat(targetVec, arrow.rotationOffset);
          arrow.offset.e() = offsetVec;
          double scale = std::abs(targetVelocity_[2]);
          arrow.scale.e().setConstant(scale * 0.2);

          currentLinVelVec.e() << 0., 0., gv_[5];
          scale = currentLinVelVec.e().tail(1).norm();
          currentLinVelVec.e().normalize();
          raisim::zaxisToRotMat(currentLinVelVec, arrowCurrentYawRate.rotationOffset);
          arrowCurrentYawRate.offset.e() = offsetVec;
          arrowCurrentYawRate.scale.e().setConstant(scale * 0.2);
        }

        arrow.setPosition(arrow.offset);
        arrow.setOrientation(arrow.rotationOffset);
        arrowCurrentYawRate.setPosition(arrowCurrentYawRate.offset);
        arrowCurrentYawRate.setOrientation(arrowCurrentYawRate.rotationOffset);
      }

      /// Update reward logger
      gui::rewardLogger.log("torqueReward", torqueReward_);
      gui::rewardLogger.log("targetFollowingReward", targetFollowingReward_);
      gui::rewardLogger.log("actionSmoothnessReward", actionSmoothnessReward_);
      gui::rewardLogger.log("bodyOrientationReward", bodyOrientationReward_);
      gui::rewardLogger.log("symmetryReward", symmetryReward_);
      gui::rewardLogger.log("footClearanceReward", footClearanceReward_);
      gui::rewardLogger.log("gaitFollowingReward", gaitFollowingReward_);
      gui::rewardLogger.log("verticalVelocityReward", verticalVelocityReward_);
      gui::rewardLogger.log("bodyRatesReward", bodyRatesReward_);
      gui::rewardLogger.log("bodyHeightReward", bodyHeightReward_);
      gui::rewardLogger.log("internalContactReward", internalContactReward_);
      gui::rewardLogger.log("jointVelocityReward", jointVelocityReward_);

      raisim::gui::jointSpeedTorqueLogger.push_back(worldTimeForVis_, gaitParams_.footTarget, gaitParams_.footPosition);
      worldTimeForVis_ += world_->getTimeStep();

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(spotVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }
  }

  void getFootContacts() {
    // Get the foot contact states
    std::fill(std::begin(gaitParams_.footContactStates), std::end(gaitParams_.footContactStates), false);
    for (auto &contact : spot_->getContacts()) {      
      if (footIndices_.find(contact.getlocalBodyIndex()) != footIndices_.end()) {
        auto footInContactIdx = std::distance(footIndices_.begin(), footIndices_.find(contact.getlocalBodyIndex()));
        gaitParams_.footContactStates[footInContactIdx] = true;
      }
    }
  }

  void updateExtraInfo() {
    // extraInfo_["forward vel reward"] = forwardVelReward_;
    // extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    spot_->getState(gc_, gv_);
    getFootContacts();

    /// body height
    auto frame = spot_->getFrameByName("ROOT");
    raisim::Vec<3> basePosition;
    spot_->getFramePosition(frame, basePosition);
    bodyHeight_ = gc_[2] - heightMap_->getHeight(basePosition[0], basePosition[1]);

    int pos = 0;
    // obDouble_[pos] = bodyHeight_; pos++;

    /// body orientation
    raisim::Vec<4> quat;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, Rb_);
    obDouble_.segment(pos, 3) = Rb_.e().row(2); pos += 3;

    /// joint angles
    obDouble_.segment(pos, 12) = gc_.tail(12); pos += 12;

    /// body velocities
    bodyLinearVel_ = Rb_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = Rb_.e().transpose() * gv_.segment(3, 3);
    obDouble_.segment(pos, 3) = bodyLinearVel_; pos += 3;
    obDouble_.segment(pos, 3) = bodyAngularVel_; pos += 3;

    /// joint velocities
    obDouble_.segment(pos, 12) = gv_.tail(12);

    // for(int i = 0; i < pos; i++){
    //   obDouble_(i) += (obNoise_(i) * sampleUniform(-1., 1.));
    // }

    pos += 12;
    /// target velocity
    obDouble_.segment(pos, 3) = targetVelocity_; pos += 3;

    /// action history
    obDouble_.segment(pos, 3*actionDim_) = actionHistory_; pos += 3*actionDim_;

    /// gait parameters
    obDouble_(pos) = gaitParams_.stride; pos++;

    double mapHeightAtFoot = 0.;
    for(int i = 0; i < 4; i++){
      obDouble_(pos) = gaitParams_.swingStart[i];
      obDouble_(pos + 4) = gaitParams_.swingDuration[i];
      obDouble_(pos + 8) = gaitParams_.phaseTimeLeft[i];
      obDouble_(pos + 12) = static_cast<double>(gaitParams_.desiredContactStates[i]);
      auto frame = spot_->getFrameByName(gaitParams_.eeFrameNames[i]);
      raisim::Vec<3> footPosition;
      spot_->getFramePosition(frame, footPosition);
      mapHeightAtFoot = footPosition[2] - heightMap_->getHeight(footPosition[0], footPosition[1]);
      gaitParams_.footPosition[i] = mapHeightAtFoot;
      // obDouble_(pos + 16) = gaitParams_.footTarget[i];
      pos++;
    }
    pos += 3*4; // 4*4;

    if(step_ % bufferStride_ == 0){
      step_ = 0;
      Eigen::VectorXd temp;
      temp.setZero((bufferLength_ - 1) * nJoints_);
      temp = obDouble_.segment(pos + nJoints_, nJoints_ * (bufferLength_ - 1));
      obDouble_.segment(pos + nJoints_ * (bufferLength_ - 1), nJoints_) = pTarget_.tail(nJoints_) - gc_.tail(12); // obDouble_.segment(3, nJoints_);
      obDouble_.segment(pos, nJoints_ * (bufferLength_ - 1)) = temp;
      pos += nJoints_ * bufferLength_;

      temp = obDouble_.segment(pos + nJoints_, nJoints_ * (bufferLength_ - 1));
      obDouble_.segment(pos + nJoints_ * (bufferLength_ - 1), nJoints_) = gv_.tail(12); // obDouble_.segment(21, nJoints_);
      obDouble_.segment(pos, nJoints_ * (bufferLength_ - 1)) = temp;
      pos += nJoints_ * bufferLength_;

      temp.setZero(3 * (bufferLength_ - 1));
      temp = obDouble_.segment(pos + 3, 3 * (bufferLength_ - 1));
      Eigen::Vector3d bodyVelErr = targetVelocity_ - bodyLinearVel_;
      bodyVelErr(2) = targetVelocity_(2) - bodyAngularVel_(2);
      obDouble_.segment(pos + 3 * (bufferLength_ - 1), 3) = bodyVelErr;
      obDouble_.segment(pos, 3 * (bufferLength_ - 1)) = temp;
      pos += 3 * bufferLength_;
    }
    step_++;

    obScaled_ = (obDouble_-obMean_).cwiseQuotient(obStd_);
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obScaled_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: spot_->getContacts()){
      if(!contact.isSelfCollision()){ // filter out internal contacts
        if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
          if(contact.isObjectA()){
            return true;
          }
        }
      }
    }

    terminalReward = 0.f;
    return false;
  }

  raisim::HeightMap* generateTerrain(const YAML::Node& cfg){

    // read terrain parameters from YAML file
    auto terrainNode = cfg["terrain"];
    READ_YAML(double, terrainProperties_.frequency, terrainNode["frequency"])
    READ_YAML(double, terrainProperties_.zScale, terrainNode["zScale"])
    READ_YAML(double, terrainProperties_.xSize, terrainNode["xSize"])
    READ_YAML(double, terrainProperties_.ySize, terrainNode["ySize"])
    READ_YAML(int, terrainProperties_.xSamples, terrainNode["xSamples"])
    READ_YAML(int, terrainProperties_.ySamples, terrainNode["ySamples"])
    READ_YAML(int, terrainProperties_.fractalOctaves, terrainNode["fractalOctaves"])
    READ_YAML(double, terrainProperties_.fractalLacunarity, terrainNode["fractalLacunarity"])
    READ_YAML(double, terrainProperties_.fractalGain, terrainNode["fractalGain"])

    // randomize the terrain
    terrainProperties_.frequency = sampleUniform(terrainProperties_.frequency * 0.5, terrainProperties_.frequency * 1.5);


    return world_->addHeightMap(0.0, 0.0, terrainProperties_);

  }

  void setSeed(int seed) final {
    std::srand(seed);
  }

  void close() final {
  }

  inline double sampleUniform(double lower, double upper) {
    return lower + uniformDist_(randNumGen_) * (upper - lower);
  }

  inline double wrap_01(double a) {
    return a - fastFloor(a);
  }

  inline int fastFloor(double a) {
    int i = int(a);
    if (i > a) i--;
    return i;
  }

  template<typename T>
  inline void getRangeFromYaml(std::vector<T> &out, YAML::Node in) {
    if (in.size() == 2){
      if (in[0] > in[1]){
        std::cout << "[Environment.hpp/getRangeFromYaml(...)] Array value at index 1 must be greater than the value at index 0." << std::endl;
        throw;
      }
    }
    out = in.template as<std::vector<T>>();
  }

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  std::normal_distribution<double> distribution_;
  raisim::ArticulatedSystem* spot_;
  std::vector<GraphicObject> * spotVisual_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_;
  double terminalRewardCoeff_ = -10.;
  double targetFollowingRewardCoeff_ = 0., actionSmoothnessRewardCoeff_ = 0., bodyOrientationRewardCoeff_ = 0., symmetryRewardCoeff_ = 0., footClearanceRewardCoeff_ = 0., gaitFollowingRewardCoeff_ = 0., verticalVelocityRewardCoeff_ = 0., bodyRatesRewardCoeff_ = 0., bodyHeightRewardCoeff_ = 0., internalContactRewardCoeff_ = 0., jointVelocityRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0., targetFollowingReward_ = 0., actionSmoothnessReward_ = 0., bodyOrientationReward_ = 0., symmetryReward_ = 0., footClearanceReward_ = 0., gaitFollowingReward_ = 0., verticalVelocityReward_ = 0., bodyRatesReward_ = 0., bodyHeightReward_ = 0., internalContactReward_ = 0., jointVelocityReward_ = 0.;
  double desired_fps_ = 60.;
  int visualizationCounter_=0;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_, obNoise_;
  Eigen::VectorXd obDouble_, obScaled_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, targetVelocity_;
  std::set<size_t> footIndices_;
  std::mt19937 randNumGen_;
  std::uniform_real_distribution<double> uniformDist_;
  std::map<std::string, std::vector<double>> commandParams_;
  Eigen::VectorXd actionHistory_;
  raisim::Mat<3,3> Rb_; // rotation matrix from the base to inertial frame
  double bodyHeight_ = 0.54;
  double bodyHeightTarget_ = 0.54;
  double worldTimeForVis_ = 0.;
  double gaitFreq_ = 0.;
  int bufferLength_ = 1;
  int bufferStride_ = 1;
  int step_ = -1;
  
  typedef struct gaitParams{
    double stride = 0.8; // [s]
    double maxFootHeight = 0.17; // [m]
    double phase = 0.;
    double swingStart[4] = {0.0, 0.5, 0.5, 0.0}; // [s]
    double swingDuration[4] = {0.5, 0.5, 0.5, 0.5}; // [s]
    double phaseTimeLeft[4] = {1.0, 1.0, 1.0, 1.0};
    double footTarget[4] = {0.17, 0.17, 0.17, 0.17}; // [m]
    double footPosition[4] = {0., 0., 0., 0.}; // [m]
    bool isStanceGait = false;
    std::array<bool,4> footContactStates = {true, true, true, true};
    std::array<bool,4> desiredContactStates = {true, true, true, true};
    std::string eeFrameNames[4] = {"front_left_foot", "front_right_foot", "rear_left_foot", "rear_right_foot"};
  };

  gaitParams gaitParams_;
  gaitParams defaultGaitParams_;
  gaitParams stanceGaitParams_;

  // terrain generation
  raisim::TerrainProperties terrainProperties_;
  raisim::HeightMap* heightMap_;

  Eigen::VectorXd genForce_, pGain_, iGain_, dGain_, errCurr_, errPrev_, errInt_;

};

}