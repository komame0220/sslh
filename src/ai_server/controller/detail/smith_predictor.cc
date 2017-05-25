#include <cmath>
#include "smith_predictor.h"

namespace ai_server {
namespace controller {
namespace detail {

smith_predictor::smith_predictor(const double cycle, const double zeta, const double omega)
    : cycle_(cycle), zeta_(zeta), omega_(omega) {
  for (int i = 0; i < 7; i++) {
    u_[i] = {0.0, 0.0, 0.0};
  }
}

Eigen::Matrix3d smith_predictor::interpolate(const model::robot& robot, const velocity_t u) {
  u_[0] = u; // 受け取った制御入力を最新入力として

  // 受け取ったロボットの状態(無駄時間分の遅れ含む)
  // 計算しやすいように,ベクトルに変換
  Eigen::Vector3d pre_position(robot.x(), robot.y(), robot.theta());
  Eigen::Vector3d pre_velocity(robot.vx(), robot.vy(), robot.omega());
  Eigen::Vector3d pre_acceleration(robot.ax(), robot.ay(), robot.alpha());

  Eigen::Vector3d now_position     = pre_position;
  Eigen::Vector3d now_velocity     = pre_velocity;
  Eigen::Vector3d now_acceleration = pre_acceleration;

  // 1ループ毎に当時の制御入力によって1フレーム分の補間をする
  for (int i = 6; i >= 0; i--) {
    Eigen::Vector3d u(u_[i].vx, u_[i].vy, u_[i].omega);
    // p=p+v*dt
    now_position = now_position + cycle_ * pre_velocity;
    // v=v+a*dt
    now_velocity = now_velocity + cycle_ * pre_acceleration;
    // a=a+omega^2*v*dt-2*zeta*omega*a+omega^2*u
    now_acceleration =
        now_acceleration +
        cycle_ * (-std::pow(omega_, 2) * pre_velocity - 2 * zeta_ * omega_ * pre_acceleration +
                  std::pow(omega_, 2) * u);

    // 更新
    if (i != 0) {
      u_[i] = u_[i - 1];
    }
    pre_position     = now_position;
    pre_velocity     = now_velocity;
    pre_acceleration = now_acceleration;
  }

  // 返すために変換
  Eigen::Matrix3d now_state;
  now_state.block(0, 0, 3, 1) = now_position;
  now_state.block(0, 1, 3, 1) = now_velocity;
  now_state.block(0, 2, 3, 1) = now_acceleration;

  return now_state;
}

} // detail
} // controller
} // ai_serever