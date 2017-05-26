#include <cmath>

#include "ai_server/game/action/receive.h"
#include "ai_server/util/math/to_vector.h"
#include "ai_server/util/math.h"

namespace ai_server {
namespace game {
namespace action {
void receive::set_dribble(int dribble) {
  dribble_ = dribble;
}
int receive::dribble() {
  return dribble_;
}
model::command receive::execute() {
  //それぞれ自機を生成
  model::command command(id_);

  //ドリブルさせる
  command.set_dribble(dribble_);

  const auto& robots = is_yellow_ ? world_.robots_yellow() : world_.robots_blue();
  if (!robots.count(id_)) {
    command.set_velocity({0.0, 0.0, 0.0});
    return command;
  }

  const auto& robot      = robots.at(id_);
  const auto robot_pos   = util::math::position(robot);
  const auto robot_theta = util::wrap_to_pi(robot.theta());
  const auto ball_pos    = util::math::position(world_.ball());
  const auto ball_vec    = util::math::velocity(world_.ball());

  //ボールがめっちゃ近くに来たら受け取ったと判定
  //現状だとボールセンサに反応があるか分からないので
  if ((robot_pos - ball_pos).norm() < 120) {
    flag_ = true;
    command.set_velocity({0.0, 0.0, 0.0});
    return command;
  }
  const auto length   = robot_pos - ball_pos;
  const auto normaliz = ball_vec.normalized();
  const auto dot      = normaliz.dot(length);
  //目標位置と角度
  const auto target = (ball_pos + dot * normaliz);
  const auto theta  = std::atan2(-normaliz.y(), -normaliz.x());

  const auto omega = theta - robot_theta;
  command.set_position({target.x(), target.y(), omega});

  flag_ = false;
  return command;
}
bool receive::finished() const {
  return flag_;
}
}
}
}
