#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <cmath>

#include "ai_server/util/math.h"
#include "state_feedback_controller.h"

namespace ai_server {
namespace controller {

using boost::math::constants::half_pi;
using boost::math::constants::pi;

const double state_feedback_controller::k_         = 49.17;
const double state_feedback_controller::zeta_      = 1.0;
const double state_feedback_controller::omega_     = 49.17;
const double state_feedback_controller::v_max_     = 10000.0;
const double state_feedback_controller::omega_max_ = 2.0;

state_feedback_controller::state_feedback_controller(const double cycle)
    : base(v_max_),
      cycle_(cycle),
      velocity_generator_(cycle_),
      smith_predictor_(cycle_, zeta_, omega_) {
  // 状態フィードバックゲイン
  // (s+k)^2=s^2+2ks+k^2=0
  // |sI-A|=s^2+(2ζω-k2ω^2)s+ω^2-k1ω^2
  // 2k=2ζω-k2ω^2,k^2=ω^2-k1ω^2
  double k1 = 1.0 - std::pow(k_, 2) / std::pow(omega_, 2);
  double k2 = 2.0 * (zeta_ * omega_ - k_) / std::pow(omega_, 2);
  kp_       = {k1, k1, 0.0};
  ki_       = {0.0, 0.0, 0.0};
  kd_       = {k2, k2, 0.0};
  for (int i = 0; i < 2; i++) {
    up_[i] = Eigen::Vector3d::Zero();
    ui_[i] = Eigen::Vector3d::Zero();
    ud_[i] = Eigen::Vector3d::Zero();
    u_[i]  = Eigen::Vector3d::Zero();
    e_[i]  = Eigen::Vector3d::Zero();
  }
}

void state_feedback_controller::set_velocity_limit(const double limit) {
  base::set_velocity_limit(std::min(limit, v_max_));
}

velocity_t state_feedback_controller::update(const model::robot& robot,
                                             [[maybe_unused]] const model::field& field,
                                             const position_t& setpoint) {
  calculate_regulator(robot);
  Eigen::Vector3d set     = {setpoint.x, setpoint.y, setpoint.theta};
  Eigen::Vector3d e_p     = set - estimated_robot_.col(0);
  Eigen::Vector3d delta_p = convert(e_p, estimated_robot_(2, 0));

  Eigen::Vector3d target;
  {
    const double pre_vel = delta_p.head<2>().normalized().dot(u_[1].head<2>());
    const Eigen::Vector2d vel =
        velocity_generator_.control_pos(pre_vel, e_p.head<2>().norm(), stable_flag_) *
        delta_p.head<2>().normalized();
    target.x() = vel.x();
    target.y() = vel.y();
    target.z() = std::clamp(4.0 * util::math::wrap_to_pi(delta_p.z()), -omega_max_, omega_max_);
  }

  calculate_output(field, target);

  return velocity_t{u_[0].x(), u_[0].y(), u_[0].z()};
}

velocity_t state_feedback_controller::update(const model::robot& robot,
                                             [[maybe_unused]] const model::field& field,
                                             const velocity_t& setpoint) {
  calculate_regulator(robot);

  Eigen::Vector3d set    = {setpoint.vx, setpoint.vy, setpoint.omega};
  Eigen::Vector3d target = convert(set, estimated_robot_(2, 0));

  const double pre_vel = target.head<2>().normalized().dot(u_[1].head<2>());
  const Eigen::Vector2d vel =
      velocity_generator_.control_vel(pre_vel, target.head<2>().norm(), stable_flag_) *
      target.head<2>().normalized();
  target.x() = vel.x();
  target.y() = vel.y();
  target.z() = std::clamp(set.z(), -omega_max_, omega_max_);

  calculate_output(field, target);

  return velocity_t{u_[0].x(), u_[0].y(), u_[0].z()};
}

void state_feedback_controller::calculate_regulator(const model::robot& robot) {
  // 前回制御入力をフィールド基準に座標変換
  double u_direction    = std::atan2(u_[1].y(), u_[1].x());
  Eigen::Vector3d pre_u = Eigen::AngleAxisd(u_direction, Eigen::Vector3d::UnitZ()) * u_[1];

  // smith_predictorでvisionの遅れ時間の補間
  estimated_robot_ = smith_predictor_.interpolate(robot, pre_u);

  // ロボット速度を座標変換
  e_[0] = convert(estimated_robot_.col(1), estimated_robot_(2, 0));

  // 双一次変換
  // s=(2/T)*(Z-1)/(Z+1)としてPIDcontrollerを離散化
  // C=Kp+Ki/s+Kds
  up_[0] = kp_.array() * e_[0].array();
  ui_[0] = cycle_ * ki_.array() * (e_[0].array() + e_[1].array()) / 2.0 + ui_[1].array();
  ud_[0] = 2.0 * kd_.array() * (e_[0].array() - e_[1].array() / cycle_ - ud_[1].array());

  u_[0] = up_[0] + ui_[0] + ud_[0];
}

Eigen::Vector3d state_feedback_controller::convert(const Eigen::Vector3d& raw,
                                                   const double robot_theta) {
  Eigen::Vector3d target = Eigen::AngleAxisd(-robot_theta, Eigen::Vector3d::UnitZ()) * raw;
  return target;
}
// 出力計算及び後処理
void state_feedback_controller::calculate_output(const model::field& field,
                                                 Eigen::Vector3d target) {
  // ロボット入力計算
  u_[0] = u_[0] + (std::pow(k_, 2) / std::pow(omega_, 2)) * target;
  if (u_[0].head<2>().norm() > velocity_limit_) {
    const Eigen::Vector2d input_vel = velocity_limit_ * u_[0].head<2>().normalized();
    u_[0].x()                       = input_vel.x();
    u_[0].y()                       = input_vel.y();
  }
  // nanが入ったら前回入力を今回値とする
  if (std::isnan(u_[0].x()) || std::isnan(u_[0].y()) || std::isnan(u_[0].z())) {
    u_[0] = u_[1];
  }

  // 速度制限
  {
    // フィールド基準の速度指令値
    const Eigen::Vector3d org_vel = convert(u_[0], -estimated_robot_(2, 0));
    if (org_vel.head<2>().norm() > 100.0) {
      // 想定加速度
      const double acc = velocity_generator_.a_max();
      // フィールド外枠から出られる距離
      constexpr double margin = 400.0;
      // フィールド基準のロボット座標
      const Eigen::Vector2d robot_pos = estimated_robot_.col(0).topRows(2);

      // 制限速度計算
      const double posi_limit_vx =
          std::sqrt(2.0 * acc * (field.x_max() + margin - robot_pos.x()));
      const double nega_limit_vx =
          -std::sqrt(2.0 * acc * (field.x_min() - margin - robot_pos.x()));
      const double posi_limit_vy =
          std::sqrt(2.0 * acc * (field.y_max() + margin - robot_pos.y()));
      const double nega_limit_vy =
          -std::sqrt(2.0 * acc * (field.y_min() - margin - robot_pos.y()));

      Eigen::Vector3d vel;
      {
        vel.x() = std::clamp(org_vel.x(), nega_limit_vx, posi_limit_vx);
        vel.y() = std::clamp(org_vel.y(), nega_limit_vy, posi_limit_vy);
        if (vel.x() / org_vel.x() < vel.y() / org_vel.y()) {
          vel.y() = org_vel.y() * (vel.x() / org_vel.x());
        } else {
          vel.x() = org_vel.x() * (vel.y() / org_vel.y());
        }
      }
      vel.z() = org_vel.z();
      u_[0]   = convert(vel, estimated_robot_(2, 0));
    }
  }

  // 値の更新
  up_[1] = up_[0];
  ui_[1] = ui_[0];
  ud_[1] = ud_[0];
  u_[1]  = u_[0];
  e_[1]  = e_[0];
}

double state_feedback_controller::find_cross_point(const double x_1, const double y_2,
                                                   const double angle) {
  // 2点から直線の式を求め,交点を求める
  // (y_1,x_2はゼロ)
  double x;
  double y;
  if (std::abs(angle) == half_pi<double>()) {
    return y_2;
  }
  if (x_1 == 0.0 || y_2 == 0.0) {
    return 0.0;
  }
  x = x_1 * y_2 / (x_1 * std::tan(angle) + y_2);
  y = -x * y_2 / x_1 + y_2;

  // 原点から交点までの距離を返す
  return std::hypot(x, y);
}

} // namespace controller
} // namespace ai_server
