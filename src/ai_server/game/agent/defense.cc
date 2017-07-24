#include <cmath>
#include <map>

#include "ai_server/game/agent/defense.h"
#include "ai_server/model/command.h"
#include "ai_server/util/math/geometry.h"
#include "ai_server/util/math.h"

namespace ai_server {
namespace game {
namespace agent {

defense::defense(const model::world& world, bool is_yellow, unsigned int keeper_id,
                 const std::vector<unsigned int>& wall_ids,
                 const std::vector<unsigned int>& marking_ids)
    : base(world, is_yellow),
      keeper_id_(keeper_id),
      wall_ids_(wall_ids),
      marking_ids_(marking_ids),
      orientation_(Eigen::Vector2d::Zero()),
      mode_(defense_mode::normal_mode) {
  // actionの生成部分
  //
  // actionの初期化は一回しか行わない
  //
  //

  //キーパー用のaction
  keeper_ = std::make_shared<action::guard>(world_, is_yellow_, keeper_id_);

  //壁用のaction
  for (auto it : wall_ids_) {
    wall_.emplace_back(std::make_shared<action::guard>(world_, is_yellow_, it));
  }

  //クリア用のaction
  for (auto it : wall_ids_) {
    wall_g_.emplace_back(std::make_shared<action::get_ball>(world_, is_yellow_, it));
  }

  //マーキング用のaction
  for (auto it : marking_ids_) {
    marking_.emplace_back(std::make_shared<action::marking>(world_, is_yellow_, it));
  }
  ball_ = {0.0, 0.0};
}

void defense::set_mode(defense_mode mode) {
  mode_ = mode;
}

std::vector<std::shared_ptr<action::base>> defense::execute() {
  using boost::math::constants::pi;

  //ボールの座標
  const Eigen::Vector2d ball_vec(world_.ball().vx(), world_.ball().vy());
  const Eigen::Vector2d ball_pos(world_.ball().x(), world_.ball().y());
  const Eigen::Vector2d ball_k(ball_vec * 0.5);
  const Eigen::Vector2d ball(ball_pos + ball_k);

  //状態を遷移させるためのボールの位置
  if (ball_vec.norm() < 150.0) {
    ball_ = ball_pos;
  }
  //ボールがゴールより後ろに来たら現状維持
  if (ball.x() < world_.field().x_min() || ball.x() > world_.field().x_max()) {
    for (auto wall_it : wall_) {
      wall_it->set_halt(true);
    }
    keeper_->set_halt(true);

    std::vector<std::shared_ptr<action::base>> re_wall{
        wall_.begin(), wall_.end()}; //型を合わせるために無理矢理作り直す
    re_wall.push_back(keeper_);      //配列を返すためにキーパーを統合する

    return re_wall;
  }
  for (auto wall_it : wall_) {
    wall_it->set_halt(false);
  }
  keeper_->set_halt(false);
  //ゴールの座標
  const Eigen::Vector2d goal(world_.field().x_min(), 0.0);
  const Eigen::Vector2d goal_e(world_.field().x_max(), 0.0);

  const auto ball_theta = std::atan2(ball.y() - goal.y(), ball.x() - goal.x());

  //基準座標を求めている
  //
  //
  //

  //比
  {
    //半径
    const auto radius = 1380.0;

    const auto length = (goal - ball).norm(); //ボール<->ゴール

    const auto ratio = radius / length; //全体に対しての基準座標の比

    orientation_ = (1 - ratio) * goal + ratio * ball;
    if (std::abs(orientation_.y()) < 500) {
      orientation_.x() = -3500;
    } else {
      const auto A = (goal.y() - ball.y()) / (goal.x() - ball.x());
      const auto B = ball.y() - A * ball.x();
      const auto a = -1 * A;
      const auto b = 1;
      const auto c = -1 * B;
      Eigen::Vector2d pos_p{goal.x(), std::copysign(500, ball.y())};
      const auto d = a * pos_p.x() + b * pos_p.y() + c;
      Eigen::Vector2d pos_h{pos_p.x() - a * (d / (std::pow(a, 2) + std::pow(b, 2))),
                            pos_p.y() - b * (d / (std::pow(a, 2) + std::pow(b, 2)))};
      const auto r  = 1000.0;
      const auto hq = std::sqrt(std::pow(r, 2) - (d / (std::pow(a, 2) + std::pow(b, 2))));
      orientation_  = Eigen::Vector2d{pos_h.x() + hq * (b / std::hypot(a, b)),
                                     pos_h.y() - hq * (a / std::hypot(a, b))};
    }
  }

  //ここから壁の処理
  //
  //
  //
  //
  //
  {
    //壁のイテレータ
    auto wall_it = wall_.begin();

    std::vector<Eigen::Vector2d> target;
    //移動目標を出す
    {
      //基準点からどれだけずらすか
      auto shift_ = 0.0;
      //縄張りの大きさ
      const auto demarcation = 3500.0;
      //壁の数によってずらしていく倍率が変わるのでその倍率
      auto magnification = 0.0;

      if (wall_.size() % 2) { //奇数
        //奇数なら五稜郭っぽいことしたいね<-塹壕戦の基本！
        Eigen::Vector2d odd{Eigen::Vector2d::Zero()};
        //中央の壁は前に出ろ
        {
          const auto length = (orientation_ - ball).norm(); //ボール<->ゴール

          const auto ratio = 400 / length; //全体に対しての基準座標の比

          odd = ((1 - ratio) * orientation_ + ratio * ball);
        }
        //時々位置が反転するのでその処理
        {
          if (odd.x() < orientation_.x()) {
            const auto length = (orientation_ - ball).norm(); //ボール<->ゴール

            const auto ratio = -400 / length; //全体に対しての基準座標の比

            odd = (1 - ratio) * orientation_ + ratio * ball;
          }
        }

        //敵がめっちゃ近づいたら閉める
        if ((ball - goal).norm() < demarcation) {
          //前に出てる場合じゃねぇ
          odd = orientation_;
        }
        target.push_back(odd);
        wall_it++;
        magnification = 1.0;
        shift_        = 200;
      } else { //偶数
        magnification = 2.0;
        shift_        = 190.0;
        //敵がめっちゃ近づいたら閉める
        if ((ball - goal).norm() < demarcation) {
          shift_ = 90;
        }
      }
      for (auto shift = shift_; wall_it != wall_.end();
           shift += (shift_ * magnification), wall_it += 2) {
        const auto tmp = util::math::calc_isosceles_vertexes(ball, orientation_, shift);
        target.push_back(std::get<0>(tmp));
        target.push_back(std::get<1>(tmp));
        //もしディフェンスエリアないに入ってしまっても順番が入れ替わらないようにする
        if ((ball - goal).norm() < 1400) {
          const auto tmp = target.back();
          target.pop_back();
          auto target_it = target.end();
          target.insert(--target_it, tmp);
        }
      }
    }

    //実際にアクションを詰めて返す
    {
      const auto my_robots = is_yellow_ ? world_.robots_yellow() : world_.robots_blue();

      //距離が近い順に昇順ソート
      std::sort(wall_.begin(), wall_.end(), [&goal, &my_robots](const auto& a, const auto& b) {
        return std::atan2(my_robots.at(a->id()).y() - goal.y(),
                          my_robots.at(a->id()).x() - goal.x()) <
               std::atan2(my_robots.at(b->id()).y() - goal.y(),
                          my_robots.at(b->id()).x() - goal.x());
      });

      //距離が近い順に昇順ソート
      std::sort(target.begin(), target.end(),
                [&goal, &my_robots](const auto& a, const auto& b) {
                  return std::atan2(a.y() - goal.y(), a.x() - goal.x()) <
                         std::atan2(b.y() - goal.y(), b.x() - goal.x());
                });

      //割り当てる
      auto target_it = target.begin();
      for (auto wall_it : wall_) {
        wall_it->move_to((*target_it).x(), (*target_it).y(), ball_theta);
        if (mode_ == defense_mode::stop_mode) {
          wall_it->set_kick_type({model::command::kick_type_t::none, 0});
        } else {
          wall_it->set_kick_type({model::command::kick_type_t::chip, 255});
        }
        wall_it->set_dribble(0);
        const auto tmp =
            ball_vec
                .norm(); // (ball_vec.norm() * 1.0 < 1000.0) ? 1000.0 : ball_vec.norm() * 1.0;
        wall_it->set_magnification(tmp);
        target_it++;
      }
    }
  }
  //型を合わせるために無理矢理作り直す
  std::vector<std::shared_ptr<action::base>> re_wall{wall_.begin(), wall_.end()};

  //クリアさせる
  if (false) {
    if (!wall_ids_.empty() && mode_ != defense_mode::stop_mode) {
      if ((goal - ball_).norm() < 2000.0 && (goal - ball_).norm() > 1400 &&
          ball_vec.norm() < 200.0) { // ボールに最も近い敵ロボットを求める
        const auto my_robots = is_yellow_ ? world_.robots_yellow() : world_.robots_blue();
        const auto it        = std::min_element(
            re_wall.cbegin(), re_wall.cend(), [&my_robots, &ball](auto&& a, auto&& b) {
              const auto l1 =
                  Eigen::Vector2d{my_robots.at(a->id()).x(), my_robots.at(a->id()).y()} - ball;
              const auto l2 =
                  Eigen::Vector2d{my_robots.at(b->id()).x(), my_robots.at(b->id()).y()} - ball;
              return l1.norm() < l2.norm();
            });
        const auto id = (*it)->id();
        re_wall.erase(it);
        //クリアように追加する
        for (auto it : wall_g_) {
          if (it->id() == id) {
            // it->set_mode(true);
            re_wall.push_back(it);
          }
        }
      }
    }
  }

  //マーキングの処理.
  //
  //
  {
    std::vector<enemy> enemy_list;
    const auto enemy_robots = is_yellow_ ? world_.robots_blue() : world_.robots_yellow();
    if (!enemy_robots.empty()) {
      for (auto it : enemy_robots) {
        const Eigen::Vector2d tmp{(it.second).x(), (it.second).y()};
        if (((ball - tmp).norm() < 1000) || (tmp.x() > 0)) {
          continue;
        }
        enemy_list.emplace_back(enemy{it.first, tmp, (it.second).theta(), 0.0, 0});
      }
      {
        Eigen::Vector2d position{Eigen::Vector2d::Zero()};
        position = Eigen::Vector2d{-3500.0, std::copysign(1, ball_pos.y()) * 1500.0};
        for (auto it : enemy_robots) {
          if ((Eigen::Vector2d{(it.second).vx(), (it.second).vy()}).norm() > 3000.0 &&
              (Eigen::Vector2d{(it.second).x(), (it.second).y()} - position).norm() > 2000.0) {
          } else {
          }
        }
      }

      {
        //ボールとの距離で点数決め
        for (auto& it : enemy_list) {
          it.valuation = it.position.y();
        }

        //距離が近い順に昇順ソート
        std::sort(enemy_list.begin(), enemy_list.end(), [&ball](const enemy& a,
                                                                const enemy& b) {
          return std::signbit(ball.y()) ? a.valuation > b.valuation : a.valuation < b.valuation;
        });

        //点数の初期化
        auto point = enemy_list.size();
        for (auto& it : enemy_list) {
          it.score = point--;
        }
      }

      //点数が大きい順に並べ替える
      std::sort(enemy_list.begin(), enemy_list.end(),
                [](const enemy& a, const enemy& b) { return (a.score > b.score); });

      //パスカットのため先頭に捩じ込む
      {
        // ボールに最も近い敵ロボットを求める
        const auto it = std::min_element(
            enemy_robots.cbegin(), enemy_robots.cend(), [&ball](auto&& a, auto&& b) {
              const auto l1 = Eigen::Vector2d{a.second.x(), a.second.y()} - ball;
              const auto l2 = Eigen::Vector2d{b.second.x(), b.second.y()} - ball;
              return l1.norm() < l2.norm();
            });

        // 先頭に捩じ込む
        const auto& r = std::get<1>(*it);
        enemy_list.insert(enemy_list.begin(),
                          enemy{r.id(), Eigen::Vector2d{r.x(), r.y()}, r.theta(), 0.0,
                                static_cast<unsigned int>(enemy_list.size() + 1u)});
      }

      //近い順に割り当てる
      {
        //マーキングに割り当てられたロボットのidとactionのペア
        std::unordered_map<unsigned int, mark> mark_list;
        const auto mark_robots = is_yellow_ ? world_.robots_yellow() : world_.robots_blue();
        for (auto& it : marking_) {
          if (mark_robots.count(it->id())) {
            const mark robot{{mark_robots.at(it->id()).x(), mark_robots.at(it->id()).y()}, it};
            mark_list.insert(std::make_pair(it->id(), robot));
          }
        }

        //敵を起点として最近傍探索
        for (auto enemy_it = enemy_list.begin();
             !mark_list.empty() && enemy_it != enemy_list.end(); enemy_it++) {
          const auto it = std::min_element(mark_list.cbegin(), mark_list.cend(),
                                           [enemy_it](auto&& a, auto&& b) {
                                             const auto enemy_pos = enemy_it->position;
                                             const auto l1 = enemy_pos - a.second.position;
                                             const auto l2 = enemy_pos - b.second.position;
                                             return l1.norm() < l2.norm();
                                           });
          const auto& r = std::get<1>(*it);
          r.action->mark_robot(enemy_it->id);
          r.action->set_mode(action::marking::mark_mode::kick_block);
          r.action->set_radius(250.0);
          if (enemy_it == enemy_list.begin()) {
            r.action->set_radius(600.0);
            r.action->set_mode(game::action::marking::mark_mode::corner_block);
          }
          mark_list.erase(it);
        }
        //もしマークロボットが溢れたらこうなる
        if (!mark_list.empty()) {
          {
            //ボール<->敵<->ゴールの角度で決める
            for (auto& it : enemy_list) {
              const auto goal_theta =
                  std::atan2(goal.y() - it.position.y(), goal.x() - it.position.x());
              const auto ball_theta =
                  std::atan2(ball.y() - it.position.y(), ball.x() - it.position.x());
              it.valuation = goal_theta + ball_theta;
            }

            //角度が小さいに昇順ソート
            std::sort(enemy_list.begin(), enemy_list.end(), [](const enemy& a, const enemy& b) {
              return (a.valuation > b.valuation);
            });

            auto point = enemy_list.size();
            for (auto& it : enemy_list) {
              it.score = point--;
            }
          }
          //敵を起点として最近傍探索
          //先頭は無理矢理ねじ込んだやつだから除外
          for (auto enemy_it = enemy_list.begin() + 1;
               !mark_list.empty() && enemy_it != enemy_list.end(); enemy_it++) {
            const auto it = std::min_element(mark_list.cbegin(), mark_list.cend(),
                                             [enemy_it](auto&& a, auto&& b) {
                                               const auto enemy_pos = enemy_it->position;
                                               const auto l1 = enemy_pos - a.second.position;
                                               const auto l2 = enemy_pos - b.second.position;
                                               return l1.norm() < l2.norm();
                                             });
            const auto& r = std::get<1>(*it);
            r.action->mark_robot(enemy_it->id);
            r.action->set_mode(action::marking::mark_mode::shoot_block);
            mark_list.erase(it);
          }
        }
      }
    }
  }
  re_wall.insert(re_wall.end(), marking_.begin(),
                 marking_.end()); //配列を返すためにマーキングをを統合する

  //ここからキーパーの処理
  //
  //
  //
  {
    Eigen::Vector2d keeper(Eigen::Vector2d::Zero());
    const auto enemy_robots = is_yellow_ ? world_.robots_blue() : world_.robots_yellow();
    if (!enemy_robots.empty()) {
      //敵のシューターを線形探索する
      //ボールにもっとも近いやつがシューターだと仮定
      // ボールに最も近い敵ロボットを求める
      const auto it = std::min_element(
          enemy_robots.cbegin(), enemy_robots.cend(), [&ball](auto&& a, auto&& b) {
            const auto l1 = Eigen::Vector2d{a.second.x(), a.second.y()} - ball;
            const auto l2 = Eigen::Vector2d{b.second.x(), b.second.y()} - ball;
            return l1.norm() < l2.norm();
          });
      const auto r = std::get<1>(*it);

      switch (mode_) {
        case defense_mode::stop_mode: {
        }
        case defense_mode::normal_mode: {
          //キーパーはボールの位置によって動き方が2種類ある.
          // そもそもマーキング状態なら別の処理
          // C:ボールが縄張りに入ってきた！やばーい
          // B:ボールが自陣地なので壁の補強をしなければ
          const auto demarcation1 = 1000.0; //クリアする範囲
          const auto demarcation2 = 400.0;  //クリアする範囲
          const auto demarcation3 = 3500.0; //縄張りの大きさ

          if ((ball_ - goal).norm() < demarcation1 && (ball_ - goal).norm() > demarcation2 &&
              ball_vec.norm() < 500.0 && mode_ != defense_mode::stop_mode) {
            const auto ratio =
                40.0 / ((goal_e - ball_pos).norm() + 40.0); //ボール - 目標位置の比
            keeper = (-ratio * goal_e + 1 * ball_pos) / (1 - ratio);
            keeper_->set_kick_type({model::command::kick_type_t::chip, 255});
            keeper_->set_dribble(9);
          } else if (((ball_ - goal).norm() < demarcation3) || !marking_.empty()) { // C
            //ゴール前でディフェンスする
            const auto length = (goal - orientation_).norm(); //基準点<->ボール
            const auto ratio  = (500) / length; //全体に対してのキーパー位置の比
            keeper            = (1 - ratio) * goal + ratio * orientation_;
            keeper_->set_kick_type({model::command::kick_type_t::none, 0});
            keeper_->set_dribble(0);
          } else { // B*/
            keeper_->set_kick_type({model::command::kick_type_t::none, 0});
            keeper_->set_dribble(0);
            // if (ball_vec.norm() < 500.0 && !std::signbit(ball_.x())) {
            //壁のすぐ後ろで待機
            //基準点からちょっと下がったキーパの位置
            const auto length = (goal - ball).norm(); //基準点<->ボール
            const auto ratio  = (1100) / length; //全体に対してのキーパー位置の比
            keeper            = (1 - ratio) * goal + ratio * ball;
            //            //} else {
            // const auto A = (ball.y() - ball_pos.y()) / (ball.x() - ball_pos.x());
            // const auto B = ball_pos.y() - A * ball_pos.x();
            // Eigen::Vector2d tmp_pos{goal.x(), A * world_.field().x_min() + B};
            ////Eigen::Vector2d tmp_pos{goal.x(), 0.0};
            // if (std::abs(tmp_pos.y()) > 450.0) {
            //  keeper = Eigen::Vector2d{tmp_pos.x() + 110, std::copysign(450.0, tmp_pos.y())};
            //} else {
            //	keeper=tmp_pos;
            //      const auto length = (tmp_pos - ball_pos).norm(); //敵機とゴールの距離
            //      const auto ratio    = (400.0) / length;
            //      const auto position = (1 - ratio) * tmp_pos + ratio * ball_pos;
            //	keeper=position;
            ////内積を使って導出
            ////ボールの速度を正規化
            // const auto normalize = ball_vec.normalized();
            ////対象とreceiverの距離
            // const auto length = goal - ball_pos;
            ////内積より,対象と自分の直交する位置
            // const auto dot = normalize.dot(length);
            ////目標位置に変換
            // const auto position = (ball_pos + dot * normalize);
            // std::cout<<position.x()<<" :pos: "<<position.y()<<std::endl;
            // const auto theta1   = util::wrap_to_2pi(acos((position - goal).norm() / 500.0));
            // const auto theta2 =
            //    util::wrap_to_2pi(acos((position - goal).norm() / (tmp_pos - goal).norm()));
            // keeper = Eigen::Vector2d{500 * std::sin(theta1 + theta2) + goal.x(),
            //                                       500 * std::cos(theta1 + theta2) +
            //                                       goal.y()};
            //}
            //}
          }
          const auto tmp = (ball_vec.norm() * 1.0 < 1000.0) ? 1000.0 : ball_vec.norm() * 1.0;
          keeper_->set_magnification(tmp);
          break;
        }
        case defense_mode::pk_normal_mode: {
          //キーパーのy座標は敵シューターの視線の先
          keeper.y() = (1000.0 * std::tan(r.theta())) * (-1);

          //ゴールの範囲を超えたら跳びでないようにする
          if (keeper.y() > 410) {
            keeper.y() = 410;
          } else if (keeper.y() < -410) {
            keeper.y() = -410;
          }

          //ロボットの大きさ分ずらす
          keeper.x() = goal.x() + 70.0;

          keeper_->set_magnification(3500.0);
          keeper_->set_dribble(0);
          break;
        }
        case defense_mode::pk_extention_mode: {
          keeper = ball_pos;
          keeper.x() += 45.0;
          keeper_->set_magnification(6000.0);
          keeper_->set_dribble(9);
          break;
        }
      }
      keeper_->move_to(keeper.x(), keeper.y(), ball_theta);
      re_wall.push_back(keeper_); //配列を返すためにキーパーを統合する
    }
  }

  return re_wall; //返す
}
}
}
}
