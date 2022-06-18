#include <algorithm>
#include <atomic>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>

#include <boost/asio.hpp>
#include <boost/math/constants/constants.hpp>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gtkmm.h>

#ifdef AI_SERVER_HAS_NNABLA_EXT_CUDA
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/cuda/init.hpp>
#endif

#include "ai_server/controller/state_feedback.h"
#include "ai_server/driver.h"
#include "ai_server/filter/state_observer/ball.h"
#include "ai_server/filter/state_observer/robot.h"
#include "ai_server/filter/va_calculator.h"
#include "ai_server/game/context.h"
#include "ai_server/game/captain/first.h"
#include "ai_server/game/nnabla.h"
#include "ai_server/logger/formatter.h"
#include "ai_server/logger/logger.h"
#include "ai_server/logger/sink/function.h"
#include "ai_server/logger/sink/ostream.h"
#include "ai_server/model/refmessage_string.h"
#include "ai_server/model/team_color.h"
#include "ai_server/model/world.h"
#include "ai_server/model/updater/refbox.h"
#include "ai_server/model/updater/world.h"
#include "ai_server/radio/connection/serial.h"
#include "ai_server/radio/connection/udp.h"
#include "ai_server/radio/grsim.h"
#include "ai_server/radio/humanoid.h"
#include "ai_server/receiver/refbox.h"
#include "ai_server/receiver/robot.h"
#include "ai_server/receiver/vision.h"
#include "ai_server/util/math/affine.h"
#include "ai_server/util/math/angle.h"
#include "ai_server/util/thread.h"
#include "ai_server/util/time.h"

using namespace std::chrono_literals;
using namespace std::string_literals;

namespace controller = ai_server::controller;
namespace filter     = ai_server::filter;
namespace game       = ai_server::game;
namespace logger     = ai_server::logger;
namespace model      = ai_server::model;
namespace radio      = ai_server::radio;
namespace receiver   = ai_server::receiver;
namespace util       = ai_server::util;

// 60fpsの時にn framesにかかる時間を表現する型
using fps60_type =
    std::chrono::duration<std::chrono::steady_clock::time_point::rep, std::ratio<1, 60>>;

// 攻撃方向
enum class dir { right, left };

// 基本設定
// --------------------------------

// チームの最大のロボット数 (ID 0 ~ ID max_robots - 1 の機体を使う)
static constexpr std::size_t max_robots = 16;

// WorldModelの設定
static constexpr auto use_va_filter = true; // ロボットの速度/加速度を計算する
static constexpr auto use_ball_observer = true; // ボールの状態オブザーバを有効にする

// ロボットの状態オブザーバの設定
static constexpr auto use_robot_observer = false; // 自チームのロボットでは状態オブザーバを使う
static constexpr auto lost_duration = 1s; // ロスト判定するまでの時間

// Visionの設定
static constexpr char vision_address[] = "224.5.23.2";
static constexpr short vision_port     = 10020;
static constexpr int num_cameras       = 8;

// Refboxの設定
static constexpr char refbox_address[] = "224.5.23.1";
static constexpr short refbox_port     = 10003;

// Robotの設定
static constexpr char robot_address[] = "224.5.23.2";
static constexpr short robot_port     = 10004;

// Radioの設定
static constexpr bool is_grsim            = true;
static constexpr bool use_udp             = true;
static constexpr char xbee_path[]         = "/dev/ttyUSB0";
static constexpr char grsim_address[]     = "127.0.0.1";
static constexpr short grsim_command_port = 20011;

// 制御周期の設定
static constexpr auto cycle =
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(fps60_type{1});

// stopgame時の速度制限
static constexpr double velocity_limit_at_stopgame = 1400.0;

// nnabla の設定
std::vector<std::string> nnabla_backend() {
#ifdef AI_SERVER_HAS_NNABLA_EXT_CUDA
  return {"cudnn", "cuda", "cpu"};
#else
  return {"cpu"};
#endif
}

std::string nnabla_device_id() {
  return "0";
}

// .nnp ファイルの設定
auto nnp_files(const std::filesystem::path& config_dir)
    -> std::unordered_map<std::string, game::nnabla::nnp_file_type> {
  const auto nnp_dir = config_dir / "nnp";
  return {
      // { key, { path, 常に CPU で計算するか } }
      {"probability", {nnp_dir / "game/detail/mcts/probability.nnp", true}},
  };
}

// スコープを抜けるときに io_context と thread を stop(), join() する helper
class stop_and_join_at_exit {
  boost::asio::io_context& ctx_;
  std::thread thread_;

public:
  stop_and_join_at_exit(const stop_and_join_at_exit&) = delete;
  stop_and_join_at_exit(stop_and_join_at_exit&&)      = delete;

  stop_and_join_at_exit(boost::asio::io_context& ctx, std::thread thread)
      : ctx_{ctx}, thread_{std::move(thread)} {}

  ~stop_and_join_at_exit() {
    ctx_.stop();
    thread_.join();
  }
};

// Gameを行うクラス
// --------------------------------
template <class UpdaterRefbox>
class game_runner {
public:
  game_runner(const std::filesystem::path& config_dir, model::updater::world& world,
              UpdaterRefbox& refbox, ai_server::driver& driver,
              std::shared_ptr<radio::base::command> radio)
      : running_{false},
        need_reset_{false},
        config_dir_{config_dir},
        team_color_{model::team_color::yellow},
        updater_world_{world},
        updater_refbox_{refbox},
        active_robots_{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u},
        driver_{driver},
        radio_{radio},
        l_{"game_runner"} {
    auto lock = driver_.lock();
    driver_.set_team_color(team_color_);

    for (auto id : active_robots_) {
      constexpr auto cycle_count = std::chrono::duration<double>(cycle).count();
      auto controller            = std::make_unique<controller::state_feedback>(cycle_count);
      driver_.register_robot(id, std::move(controller), radio_);
    }

    if constexpr (use_robot_observer) {
      reset_state_observers();
      on_command_updated_connection_ = driver_.on_command_updated([this](auto&&... args) {
        handle_command_updated(std::forward<decltype(args)>(args)...);
      });
    }
  }

  ~game_runner() {
    running_ = false;
    cv_.notify_all();
    if (game_thread_.joinable()) game_thread_.join();
  }

  void start() {
    std::unique_lock lock{mutex_};
    if (!game_thread_.joinable()) {
      running_     = true;
      game_thread_ = std::thread([this] { main_loop(); });
      util::set_thread_name(game_thread_, "game_thread");
    }
  }

  void stop() {
    running_ = false;
    cv_.notify_all();
    game_thread_.join();
  }

  std::vector<unsigned int> active_robots() const {
    std::unique_lock lock{mutex_};
    return active_robots_;
  }

  void set_active_robots(const std::vector<unsigned int>& ids) {
    auto lock1 = std::unique_lock{mutex_, std::defer_lock};
    auto lock2 = driver_.lock(std::defer_lock);
    std::lock(lock1, lock2);

    bool changed = false;
    // register new robots
    for (auto id : ids) {
      if (!driver_.registered(id)) {
        constexpr auto cycle_count = std::chrono::duration<double>(cycle).count();
        auto controller            = std::make_unique<controller::state_feedback>(cycle_count);
        driver_.register_robot(id, std::move(controller), radio_);
        if constexpr (use_robot_observer) set_state_observer(id);
        changed = true;
      }
    }
    // unregister robots
    for (auto id : active_robots_) {
      if (std::find(ids.cbegin(), ids.cend(), id) == ids.cend()) {
        driver_.unregister_robot(id);
        if constexpr (use_robot_observer) clear_state_observer(id);
        changed = true;
      }
    }
    if (changed) {
      l_.info(fmt::format("active_robots changed: {} -> {}", active_robots_, ids));

      active_robots_ = ids;
      need_reset_    = true;
    }
  }

  void set_team_color(model::team_color color) {
    auto lock1 = std::unique_lock{mutex_, std::defer_lock};
    auto lock2 = driver_.lock(std::defer_lock);
    std::lock(lock1, lock2);

    if (team_color_ != color) {
      driver_.set_team_color(color);
      team_color_ = color;
      need_reset_ = true;
      if constexpr (use_robot_observer) reset_state_observers();
    }
  }

  void set_transformation_matrix(double x, double y, double theta) {
    std::unique_lock lock{mutex_};
    const auto mat = util::math::make_transformation_matrix(x, y, theta);
    updater_world_.set_transformation_matrix(mat);
    updater_refbox_.set_transformation_matrix(mat);
  }

private:
  void main_loop() {
    l_.info("game started!");

    game::context ctx{};
    ctx.nnabla = std::make_unique<game::nnabla>(nnabla_backend(), nnabla_device_id(),
                                                nnp_files(config_dir_));

    model::refbox refbox{};
    std::unique_ptr<game::captain::base> captain{};

    std::chrono::steady_clock::time_point prev_time{};

    for (;;) {
      try {
        std::unique_lock lock{mutex_};
        // 前回の処理開始から cycle 待つ
        if (cv_.wait_until(lock, prev_time + cycle, [this] { return !running_; })) {
          break; // その間に stop() されたらループを抜ける
        }

        const auto current_time = std::chrono::steady_clock::now();

        const auto prev_cmd = refbox.command();

        ctx.team_color = team_color_;
        ctx.world      = updater_world_.value();
        refbox         = updater_refbox_.value();

        const auto current_cmd = refbox.command();

        if (current_cmd != prev_cmd || need_reset_) {
          if (current_cmd == model::refbox::game_command::stop) {
            driver_.set_velocity_limit(velocity_limit_at_stopgame);
          } else {
            driver_.set_velocity_limit(std::numeric_limits<double>::max());
          }
        }

        if (!captain || need_reset_) {
          captain = std::make_unique<game::captain::first>(
              ctx, refbox, std::set(active_robots_.cbegin(), active_robots_.cend()));
          need_reset_ = false;
          l_.info("captain resetted");
        }

        auto formation = captain->execute();
        auto actions   = formation->execute();
        for (auto action : actions) {
          auto command = action->execute();
          driver_.update_command(action->id(), command);
        }

        prev_time = current_time;
      } catch (const std::exception& e) {
        l_.error(fmt::format("exception at game_thread\n\t{}", e.what()));
      } catch (...) {
        l_.error("unknown exception at game_thread");
      }
    }

    // ロボットを全て停止させる
    {
      std::unique_lock lock{mutex_};
      for (const auto& id : active_robots_) {
        driver_.update_command(id, {});
      }
    }

    l_.info("game stopped!");
  }

  // id の state_observer を初期化する
  void set_state_observer(unsigned int id) {
    auto f = [this, id](auto& updater) {
      state_observers_[id] =
          updater.template set_filter<filter::state_observer::robot>(id, lost_duration);
    };
    if (team_color_ == model::team_color::yellow) {
      f(updater_world_.robots_yellow_updater());
    } else {
      f(updater_world_.robots_blue_updater());
    }
  }

  // state_observer を (再) 初期化する
  void reset_state_observers() {
    if (team_color_ == model::team_color::yellow) {
      updater_world_.robots_blue_updater().clear_all_filters();
    } else {
      updater_world_.robots_yellow_updater().clear_all_filters();
    }
    for (auto id : active_robots_) {
      set_state_observer(id);
    }
  }

  // id の state_observer を解除する
  void clear_state_observer(unsigned int id) {
    if (team_color_ == model::team_color::yellow) {
      updater_world_.robots_yellow_updater().clear_filter(id);
    } else {
      updater_world_.robots_blue_updater().clear_filter(id);
    }
  }

  // contrtoller が新しい値を出力したときの処理
  void handle_command_updated(model::team_color, unsigned int id,
                              const model::command::kick_flag_t&, int, double vx, double vy,
                              double) {
    if (auto p = state_observers_.at(id).lock()) {
      p->observe(vx, vy);
    } else {
      l_.warn(fmt::format("state_observer for id {} is not initialized / already dead", id));
      return;
    }
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> running_;

  // captain のリセットが必要か
  bool need_reset_;

  const std::filesystem::path& config_dir_;

  model::team_color team_color_;
  model::updater::world& updater_world_;
  UpdaterRefbox& updater_refbox_;
  std::vector<unsigned int> active_robots_;

  ai_server::driver& driver_;

  std::shared_ptr<radio::base::command> radio_;

  std::thread game_thread_;

  boost::signals2::connection on_command_updated_connection_;
  std::array<std::weak_ptr<filter::state_observer::robot>, max_robots> state_observers_;

  logger::logger l_;
};

// vision の設定を行うパネル
class vision_panel final : public Gtk::Frame {
public:
  using signal_switch_changed_type = sigc::signal<void(bool)>;

  signal_switch_changed_type signal_switch_changed() {
    return signal_switch_changed_;
  }

  vision_panel() : Gtk::Frame{"Vision"}, l1_{"Viewer", Gtk::ALIGN_END} {
    sw_.set_active(true);
    sw_.set_halign(Gtk::ALIGN_START);
    sw_.property_active().signal_changed().connect(
        [this] { signal_switch_changed_.emit(sw_.get_state()); });

    grid_.set_border_width(4);
    grid_.set_row_spacing(4);
    grid_.set_column_spacing(8);
    grid_.set_column_homogeneous(true);

    grid_.attach(l1_, 1, 1);

    grid_.attach_next_to(sw_, l1_, Gtk::POS_RIGHT, 2, 1);

    this->add(grid_);
  }

private:
  Gtk::Grid grid_;
  Gtk::Label l1_;
  Gtk::Switch sw_;
  signal_switch_changed_type signal_switch_changed_;
};

// vision からの位置情報を描画するエリア
class vision_area final : public Gtk::DrawingArea {
public:
  using signal_ball_position_changed_type = sigc::signal<void, double /* x */, double /* y */>;
  using signal_abp_target_changed_type    = sigc::signal<void, double /* x */, double /* y */>;
  using signal_robot_position_changed_type =
      sigc::signal<void, model::team_color, unsigned int /* id */, double /* x */,
                   double /* y */, double /* theta */>;

  signal_ball_position_changed_type signal_ball_position_changed() {
    return signal_ball_position_changed_;
  }

  signal_abp_target_changed_type signal_abp_target_changed() {
    return signal_abp_target_changed_;
  }

  signal_robot_position_changed_type signal_robot_position_changed() {
    return signal_robot_position_changed_;
  }

  vision_area(model::updater::world& world)
      : updater_world_(world),
        i1_("Locate ball here"),
        i2_("Set abp_target here"),
        i3_("Locate yellow robot here"),
        i4_("Locate blue robot here") {
    this->add_tick_callback([this](const Glib::RefPtr<Gdk::FrameClock>&) {
      this->queue_draw();
      return true;
    });

    this->add_events(Gdk::BUTTON_PRESS_MASK);
    this->signal_button_press_event().connect(
        sigc::mem_fun(*this, &vision_area::handle_set_position));

    i1_.signal_activate().connect([this] {
      // signal_ball_position_changed を発火する
      const auto& p = position_to_set_;
      signal_ball_position_changed_.emit(p.x(), p.y());
    });
    i2_.signal_activate().connect([this] {
      // signal_abp_target_changed を発火させる
      const auto& p = position_to_set_;
      signal_abp_target_changed_.emit(p.x(), p.y());
    });
    i3_.set_submenu(yellow_menu_);
    i4_.set_submenu(blue_menu_);

    for (std::size_t i = 0; i < yellow_items_.size(); ++i) {
      yellow_items_.at(i).signal_activate().connect([this, i] {
        // signal_robot_position_changed を発火させる
        const auto& p = position_to_set_;
        signal_robot_position_changed_.emit(model::team_color::yellow, i, p.x(), p.y(), 0.0);
      });
      yellow_items_.at(i).set_label(std::to_string(i));
      yellow_menu_.append(yellow_items_.at(i));
    }
    for (std::size_t i = 0; i < blue_items_.size(); ++i) {
      blue_items_.at(i).signal_activate().connect([this, i] {
        // signal_robot_position_changed を発火させる
        const auto& p = position_to_set_;
        signal_robot_position_changed_.emit(model::team_color::blue, i, p.x(), p.y(), 0.0);
      });
      blue_items_.at(i).set_label(std::to_string(i));
      blue_menu_.append(blue_items_.at(i));
    }

    menu_.append(i1_);
    menu_.append(i2_);
    menu_.append(i3_);
    menu_.append(i4_);
    menu_.attach_to_widget(*this);
    menu_.show_all();
  }

private:
  bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr) override {
    const auto allocation = get_allocation();
    const auto width      = allocation.get_width();
    const auto height     = allocation.get_height();
    const auto world      = updater_world_.value();
    const auto wf         = world.field();

    constexpr auto line_width   = 10.0;
    constexpr auto field_margin = 400.0;
    const auto game_width       = 2.0 * wf.x_max();
    const auto game_height      = 2.0 * wf.y_max();
    const auto field_width      = game_width + 2.0 * field_margin;
    const auto field_height     = game_height + 2.0 * field_margin;
    const auto penalty_width    = wf.penalty_length();
    const auto penalty_height   = wf.penalty_width();
    constexpr auto goal_width   = 200.0;
    const auto goal_height      = wf.goal_width();
    constexpr auto robot_rad    = 90.0;
    constexpr auto ball_rad     = 21.0;

    // フィールドを縦に表示するか
    const bool is_vertical = width < height;
    if (is_vertical) {
      // エリアが縦長なら90度回転
      const auto coef = std::min(height / field_width, width / field_height);
      matrix_         = Cairo::Matrix{
          0, -coef, -coef, 0, coef * field_height / 2.0, coef * field_width / 2.0};
    } else {
      const auto coef = std::min(height / field_height, width / field_width);
      matrix_ =
          Cairo::Matrix{coef, 0, 0, -coef, coef * field_width / 2.0, coef * field_height / 2.0};
    }
    cr->transform(matrix_);

    // フィールド
    {
      // フィールド全体
      cr->rectangle(-field_width / 2.0, -field_height / 2.0, field_width, field_height);
      cr->set_source_rgb(0.17, 0.17, 0.18);
      cr->fill();

      // ラインの幅と色をまとめて設定
      cr->set_source_rgb(1.0, 1.0, 1.0);
      cr->set_line_width(line_width);

      // ゲームエリアの枠
      cr->rectangle(-game_width / 2.0, -game_height / 2.0, game_width, game_height);
      cr->stroke();
      // センターライン
      cr->move_to(0.0, game_height / 2.0);
      cr->line_to(0.0, -game_height / 2.0);
      cr->stroke();
      // センターサークル
      cr->arc(0.0, 0.0, wf.center_radius(), 0.0, boost::math::double_constants::two_pi);
      cr->stroke();
      // 左ペナルティエリア
      cr->rectangle(-game_width / 2.0, -penalty_height / 2.0, penalty_width, penalty_height);
      cr->stroke();
      // 右ペナルティエリア
      cr->rectangle(game_width / 2.0 - penalty_width, -penalty_height / 2.0, penalty_width,
                    penalty_height);
      cr->stroke();
      // ペナルティマーク
      cr->arc(wf.back_penalty_mark().x, wf.back_penalty_mark().y, ball_rad, 0.0,
              boost::math::double_constants::two_pi);
      cr->stroke();
      // ペナルティマーク
      cr->arc(wf.front_penalty_mark().x, wf.front_penalty_mark().y, ball_rad, 0.0,
              boost::math::double_constants::two_pi);
      cr->stroke();
      // 左ゴール
      cr->rectangle(-game_width / 2.0 - goal_width, -goal_height / 2.0, goal_width,
                    goal_height);
      cr->stroke();
      // 右ゴール
      cr->rectangle(game_width / 2.0, -goal_height / 2.0, goal_width, goal_height);
      cr->stroke();
    }

    // ロボット
    auto draw_robots = [&cr, is_vertical](const model::world::robots_list& robots,
                                          model::team_color color) {
      if (color == model::team_color::yellow) {
        cr->set_source_rgb(1.0, 0.84, 0.04);
      } else if (color == model::team_color::blue) {
        cr->set_source_rgb(0.04, 0.52, 1.0);
      } else {
        cr->set_source_rgb(1.0, 1.0, 1.0);
      }
      for (const auto& [id, robot] : robots) {
        cr->arc(robot.x(), robot.y(), robot_rad, util::math::wrap_to_2pi(robot.theta() + 0.5),
                util::math::wrap_to_2pi(robot.theta() - 0.5));
        cr->close_path();
        cr->fill();
      }

      cr->set_source_rgb(1.0, 1.0, 1.0);
      constexpr auto font_size = 2.0 * robot_rad;
      if (is_vertical) {
        cr->set_font_matrix(
            Cairo::Matrix{0.0, -font_size, -font_size, 0.0, font_size, font_size / 4.0});
      } else {
        cr->set_font_matrix(
            Cairo::Matrix{font_size, 0.0, 0.0, -font_size, -font_size / 4.0, font_size});
      }
      for (const auto& [id, robot] : robots) {
        cr->set_source_rgb(1.0, 1.0, 1.0);
        cr->move_to(robot.x(), robot.y());
        cr->text_path(std::to_string(id));
        cr->fill();

        cr->set_line_width(20.0);
        cr->set_source_rgb(0.17, 0.17, 0.18);
        cr->move_to(robot.x(), robot.y());
        cr->line_to(robot.x() + robot_rad * std::cos(robot.theta()),
                    robot.y() + robot_rad * std::sin(robot.theta()));
        cr->stroke();
      }
    };
    draw_robots(world.robots_yellow(), model::team_color::yellow);
    draw_robots(world.robots_blue(), model::team_color::blue);

    // ボール
    {
      const auto ball = world.ball();
      cr->set_source_rgb(1.0, 0.62, 0.04);
      cr->arc(ball.x(), ball.y(), ball_rad, 0.0, boost::math::double_constants::two_pi);
      cr->fill();

      // 半径 500 [mm] のサークル
      cr->set_line_width(10.0);
      cr->set_source_rgb(1.0, 0.27, 0.23);
      cr->arc(ball.x(), ball.y(), 500.0, 0.0, boost::math::double_constants::two_pi);
      cr->stroke();
    }

    return true;
  }

  bool handle_set_position(GdkEventButton* event) {
    // 右クリックならメニュー表示
    if (event->button != 3) return false;

    // マウス座標取得
    int mouse_x, mouse_y;
    this->get_pointer(mouse_x, mouse_y);

    // 自チーム用に変換された座標をもとに戻す
    const auto x = (matrix_.yy * mouse_x - matrix_.xy * mouse_y - matrix_.yy * matrix_.x0 +
                    matrix_.xy * matrix_.y0) /
                   (matrix_.xx * matrix_.yy - matrix_.xy * matrix_.yx);
    const auto y = (-matrix_.yx * mouse_x + matrix_.xx * mouse_y + matrix_.yx * matrix_.x0 -
                    matrix_.xx * matrix_.y0) /
                   (matrix_.xx * matrix_.yy - matrix_.xy * matrix_.yx);

    // 位置をメンバに保持しておく
    position_to_set_ = util::math::transform(updater_world_.transformation_matrix().inverse(),
                                             Eigen::Vector2d{x, y});

    i1_.set_sensitive(!signal_ball_position_changed_.empty());
    i2_.set_sensitive(!signal_abp_target_changed_.empty());
    i3_.set_sensitive(!signal_robot_position_changed_.empty());
    i4_.set_sensitive(!signal_robot_position_changed_.empty());
    menu_.popup(event->button, event->time);

    return true;
  }

  model::updater::world& updater_world_;
  Cairo::Matrix matrix_;
  signal_ball_position_changed_type signal_ball_position_changed_;
  signal_abp_target_changed_type signal_abp_target_changed_;
  signal_robot_position_changed_type signal_robot_position_changed_;
  Eigen::Vector2d position_to_set_;

  std::array<Gtk::MenuItem, 16> yellow_items_, blue_items_;
  Gtk::MenuItem i1_, i2_, i3_, i4_;
  Gtk::Menu yellow_menu_, blue_menu_;
  Gtk::Menu menu_;
};

// logger の出力を表示するパネル
class log_area : public Gtk::TreeView {
public:
  log_area() : tree_model_(Gtk::ListStore::create(col_)) {
    this->set_model(tree_model_);
    this->append_column("elapsed", col_.elapsed);
    this->append_column("level", col_.level);
    this->append_column("zone", col_.zone);
    this->append_column("message", col_.message);
  }

  void write(const logger::log_item& item) {
    Gtk::TreeModel::Row row = *(tree_model_->append());
    row[col_.elapsed]       = logger::format("{elapsed}", item);
    row[col_.level]         = logger::format("{level}", item);
    row[col_.zone]          = item.zone_name;
    row[col_.message]       = item.message;
  }

private:
  struct column : public Gtk::TreeModel::ColumnRecord {
    column() {
      this->add(elapsed);
      this->add(level);
      this->add(zone);
      this->add(message);
    }

    Gtk::TreeModelColumn<Glib::ustring> elapsed;
    Gtk::TreeModelColumn<Glib::ustring> level;
    Gtk::TreeModelColumn<Glib::ustring> zone;
    Gtk::TreeModelColumn<Glib::ustring> message;
  };

  column col_;
  Glib::RefPtr<Gtk::ListStore> tree_model_;
};

// game_runner などを設定するパネル
// --------------------------------
template <class Runner>
class game_panel final : public Gtk::VBox {
public:
  game_panel(model::updater::world& world, Runner& runner)
      : updater_world_(world), runner_(runner), l_{"game_panel"} {
    {
      // left widgets
      color_.set_label("Team color");
      color_r1_.set_label("Yellow");
      color_r2_.set_label("Blue");
      color_r2_.join_group(color_r1_);
      color_box_.set_border_width(4);
      color_box_.pack_start(color_r1_);
      color_box_.pack_start(color_r2_);
      color_.add(color_box_);
      this->pack_start(color_, Gtk::PACK_SHRINK, 4);

      dir_.set_label("Attack direction");
      dir_r1_.set_label("Right");
      dir_r2_.set_label("Left");
      dir_r2_.join_group(dir_r1_);
      dir_box_.set_border_width(4);
      dir_box_.pack_start(dir_r1_);
      dir_box_.pack_start(dir_r2_);
      dir_.add(dir_box_);
      this->pack_start(dir_, Gtk::PACK_SHRINK, 4);

      robots_.set_label("Active robots");
      robots_id_box_.set_border_width(4);
      robots_id_box_.set_max_children_per_line(6);
      for (auto i = 0u; i < max_robots; ++i) {
        robots_cb_.emplace_back(std::to_string(i));
        robots_id_box_.add(robots_cb_.back());
      }
      apply_.set_label("Apply");
      apply_.set_border_width(4);
      apply_.set_sensitive(false);
      apply_.signal_clicked().connect(
          sigc::mem_fun(*this, &game_panel::handle_change_active_robots));
      robots_box_.pack_start(robots_id_box_);
      robots_box_.pack_start(apply_);
      robots_.add(robots_box_);
      this->pack_start(robots_, Gtk::PACK_SHRINK, 4);

      camera_.set_label("Camera");
      camera_id_box_.set_border_width(4);
      camera_id_box_.set_max_children_per_line(6);
      for (auto i = 0u; i < num_cameras; ++i) {
        camera_cb_.emplace_back(std::to_string(i));
        camera_id_box_.add(camera_cb_.back());
      }
      camera_.add(camera_id_box_);
      this->pack_start(camera_, Gtk::PACK_SHRINK, 4);

      start_.set_label("start");
      start_.set_sensitive(false);
      start_.signal_clicked().connect(sigc::mem_fun(*this, &game_panel::handle_start_stop));
      this->pack_end(start_, Gtk::PACK_SHRINK);

      this->set_size_request(320, -1);
    }

    init_radio_buttons();
    init_check_buttons();
  }

  void set_ready() {
    start_.set_sensitive(true);
  }

private:
  void init_radio_buttons() {
    dir_r1_.signal_toggled().connect([this] {
      runner_.set_transformation_matrix(
          0.0, 0.0, dir_r1_.get_active() ? 0.0 : boost::math::double_constants::pi);
    });

    color_r1_.signal_toggled().connect([this] {
      if (color_r1_.get_active()) {
        runner_.set_team_color(model::team_color::yellow);
      } else {
        runner_.set_team_color(model::team_color::blue);
      }
    });
  }

  void init_check_buttons() {
    auto robots = runner_.active_robots();
    for (auto i = 0u; i < max_robots; ++i) {
      auto& cb = robots_cb_.at(i);
      cb.set_active(std::find(robots.cbegin(), robots.cend(), i) != robots.cend());
      cb.signal_toggled().connect(
          sigc::mem_fun(*this, &game_panel::handle_active_robots_changed));
    }

    for (auto i = 0u; i < num_cameras; ++i) {
      auto& cb = camera_cb_.at(i);
      cb.set_active(updater_world_.is_camera_enabled(i));
      cb.signal_toggled().connect([id = i, this] {
        auto& cb = camera_cb_.at(id);
        if (cb.get_active()) {
          updater_world_.enable_camera(id);
          l_.info(fmt::format("camera{} enabled", id));
        } else {
          updater_world_.disable_camera(id);
          l_.info(fmt::format("camera{} disabled", id));
        }
      });
    }
  }

  void handle_start_stop() {
    if (start_.get_label() == "start") {
      runner_.start();
      start_.set_label("stop");
    } else {
      runner_.stop();
      start_.set_label("start");
    }
  }

  void handle_active_robots_changed() {
    apply_.set_sensitive(true);
  }

  void handle_change_active_robots() {
    std::vector<unsigned int> robots{};
    for (auto i = 0u; i < robots_cb_.size(); ++i) {
      if (robots_cb_[i].get_active()) {
        robots.push_back(i);
      }
    }
    runner_.set_active_robots(robots);
    apply_.set_sensitive(false);
  }

  Gtk::Frame color_, dir_, robots_, camera_;
  Gtk::VBox robots_box_;
  Gtk::HBox color_box_, dir_box_;
  Gtk::RadioButton color_r1_, color_r2_, dir_r1_, dir_r2_;
  std::vector<Gtk::CheckButton> robots_cb_, camera_cb_;
  Gtk::FlowBox robots_id_box_, camera_id_box_;
  Gtk::Button apply_, start_;

  model::updater::world& updater_world_;
  Runner& runner_;

  logger::logger l_;
};

// 2つのソースを切替可能な updater 的なもの
template <class U1, class U2>
class updater_selector {
  std::atomic<bool> use_global_;
  U1& global_;
  U2& local_;

public:
  static_assert(std::is_same_v<decltype(std::declval<U1>().value()),
                               decltype(std::declval<U2>().value())>);
  using value_type = decltype(std::declval<U1>().value());

  updater_selector(U1& global, U2& local) : use_global_{true}, global_{global}, local_{local} {}

  value_type value() const {
    return use_global_ ? global_.value() : local_.value();
  }

  void set_transformation_matrix(const Eigen::Affine3d& matrix) {
    global_.set_transformation_matrix(matrix);
    local_.set_transformation_matrix(matrix);
  }

  bool use_global() const {
    return use_global_;
  }
  void use_global(bool b) {
    use_global_ = b;
  }

  const U1& global() const {
    return global_;
  }
  const U2& local() const {
    return local_;
  }
};

// global / local refbox の切り替えなどを行なうパネル
class refbox_panel final : public Gtk::Frame {
  using stage_type   = model::refbox::stage_name;
  using command_type = model::refbox::game_command;

public:
  refbox_panel(model::updater::refbox& global)
      : Gtk::Frame{"Refbox"},
        l1_{"Local Refbox", Gtk::ALIGN_END},
        l2_{"stage", Gtk::ALIGN_END},
        l3_{"command", Gtk::ALIGN_END},
        l4_{"abp_target x", Gtk::ALIGN_END},
        l5_{"abp_target y", Gtk::ALIGN_END},
        l6_{"goalie Yel | Blu", Gtk::ALIGN_END},
        updater_{global, refbox_} {
    sw_.set_active(!updater_.use_global());
    sw_.set_halign(Gtk::ALIGN_START);
    sw_.property_active().signal_changed().connect(
        sigc::mem_fun(*this, &refbox_panel::handle_switch));

    for (int i = 0; i < static_cast<int>(stage_type::num_stages); ++i) {
      stages_.append(stage_name_to_string(static_cast<stage_type>(i)).data());
    }
    stages_.set_active(static_cast<int>(refbox_.refbox.stage()));
    stages_.signal_changed().connect(sigc::mem_fun(*this, &refbox_panel::handle_value_changed));

    for (int i = 0; i < static_cast<int>(command_type::num_commands); ++i) {
      commands_.append(game_command_to_string(static_cast<command_type>(i)).data());
    }
    commands_.set_active(static_cast<int>(refbox_.refbox.command()));
    commands_.signal_changed().connect(
        sigc::mem_fun(*this, &refbox_panel::handle_value_changed));

    for (int i = 0; i < 16; ++i) {
      yellow_ids_.append(std::to_string(i));
    }
    yellow_ids_.set_active(static_cast<int>(refbox_.refbox.stage()));
    yellow_ids_.signal_changed().connect(
        sigc::mem_fun(*this, &refbox_panel::handle_value_changed));

    for (int i = 0; i < 16; ++i) {
      blue_ids_.append(std::to_string(i));
    }
    blue_ids_.set_active(static_cast<int>(refbox_.refbox.stage()));
    blue_ids_.signal_changed().connect(
        sigc::mem_fun(*this, &refbox_panel::handle_value_changed));

    const auto bpp(refbox_.refbox.ball_placement_position());
    abp_x_.set_value(bpp.x());
    abp_x_.set_range(-50000, 50000);
    abp_x_.set_increments(100, 1000);
    abp_x_.signal_changed().connect(sigc::mem_fun(*this, &refbox_panel::handle_value_changed));
    abp_y_.set_value(bpp.y());
    abp_y_.set_range(-50000, 50000);
    abp_y_.set_increments(100, 1000);
    abp_y_.signal_changed().connect(sigc::mem_fun(*this, &refbox_panel::handle_value_changed));

    button_.set_label("Apply");
    button_.set_sensitive(false);
    button_.signal_clicked().connect(sigc::mem_fun(*this, &refbox_panel::handle_apply_button));

    grid_.set_border_width(4);
    grid_.set_row_spacing(4);
    grid_.set_column_spacing(8);
    grid_.set_column_homogeneous(true);

    grid_.attach(l1_, 1, 1);
    grid_.attach_next_to(l2_, l1_, Gtk::POS_BOTTOM);
    grid_.attach_next_to(l3_, l2_, Gtk::POS_BOTTOM);
    grid_.attach_next_to(l4_, l3_, Gtk::POS_BOTTOM);
    grid_.attach_next_to(l5_, l4_, Gtk::POS_BOTTOM);
    grid_.attach_next_to(l6_, l5_, Gtk::POS_BOTTOM);
    grid_.attach_next_to(sw_, l1_, Gtk::POS_RIGHT);
    grid_.attach_next_to(stages_, l2_, Gtk::POS_RIGHT, 2, 1);
    grid_.attach_next_to(commands_, l3_, Gtk::POS_RIGHT, 2, 1);
    grid_.attach_next_to(abp_x_, l4_, Gtk::POS_RIGHT, 2, 1);
    grid_.attach_next_to(abp_y_, l5_, Gtk::POS_RIGHT, 2, 1);
    grid_.attach_next_to(yellow_ids_, l6_, Gtk::POS_RIGHT, 1, 1);
    grid_.attach_next_to(blue_ids_, yellow_ids_, Gtk::POS_RIGHT, 1, 1);
    grid_.attach_next_to(button_, l6_, Gtk::POS_BOTTOM, 3, 1);

    this->add(grid_);

    handle_switch();
  }

  void set_abp_target(double x, double y) {
    abp_x_.set_value(x);
    abp_y_.set_value(y);
  }

  auto& updater() {
    return updater_;
  }

private:
  void handle_switch() {
    const auto s = sw_.get_state();
    updater_.use_global(!s);
    l_.info(fmt::format("refbox source: {}", s ? "local" : "global"));
  }

  void handle_value_changed() {
    button_.set_sensitive(true);
  }

  void handle_apply_button() {
    std::lock_guard lock{refbox_.mutex};
    if (const auto i = stages_.get_active_row_number(); i >= 0) {
      refbox_.refbox.set_stage(static_cast<stage_type>(i));
    }
    if (const auto i = commands_.get_active_row_number(); i >= 0) {
      refbox_.refbox.set_command(static_cast<command_type>(i));
    }
    if (const auto i = yellow_ids_.get_active_row_number(); i >= 0) {
      auto team_info = refbox_.refbox.team_yellow();
      team_info.set_goalie(i);
      refbox_.refbox.set_team_yellow(team_info);
    }
    if (const auto i = blue_ids_.get_active_row_number(); i >= 0) {
      auto team_info = refbox_.refbox.team_blue();
      team_info.set_goalie(i);
      refbox_.refbox.set_team_blue(team_info);
    }
    auto& p = refbox_.abp_target;
    p       = Eigen::Vector2d(abp_x_.get_value(), abp_y_.get_value());
    refbox_.refbox.set_ball_placement_position(util::math::transform(refbox_.matrix, p));
    button_.set_sensitive(false);
  }

  Gtk::Grid grid_;
  Gtk::Label l1_, l2_, l3_, l4_, l5_, l6_;
  Gtk::Switch sw_;
  Gtk::ComboBoxText stages_, commands_, yellow_ids_, blue_ids_;
  Gtk::SpinButton abp_x_, abp_y_;
  Gtk::Button button_;

  struct refbox_wrapper {
    mutable std::mutex mutex;
    model::refbox refbox;
    Eigen::Vector2d abp_target;
    Eigen::Affine3d matrix;
    refbox_wrapper()
        : abp_target{Eigen::Vector2d::Zero()}, matrix{Eigen::Translation3d{0, 0, 0}} {}
    model::refbox value() const {
      std::lock_guard lock{mutex};
      return refbox;
    }
    void set_transformation_matrix(const Eigen::Affine3d& m) {
      std::lock_guard lock{mutex};
      matrix = m;
      refbox.set_ball_placement_position(util::math::transform(matrix, abp_target));
    }
  };
  refbox_wrapper refbox_;
  updater_selector<model::updater::refbox, refbox_wrapper> updater_;

  logger::logger_for<refbox_panel> l_;
};

// 各種ステータスを表示するパネル
class status_tree final : public Gtk::TreeView {
public:
  status_tree() : store_{Gtk::TreeStore::create(columns_)} {
    this->set_model(store_);
    this->append_column("Name", columns_.name);
    this->append_column("Value", columns_.value);
    this->set_enable_tree_lines();
  }

  struct model final : public Gtk::TreeModel::ColumnRecord {
    model() {
      add(name);
      add(value);
    }
    Gtk::TreeModelColumn<Glib::ustring> name;
    Gtk::TreeModelColumn<Glib::ustring> value;
  };

  template <class T, class = void>
  struct handler;

  template <class, class = void>
  struct has_handler : std::false_type {};

  template <class T>
  struct has_handler<
      T, std::void_t<decltype(std::string{handler<T>::category_name}),
                     decltype(std::declval<const handler<T>>().update(std::declval<T>())),
                     std::enable_if_t<std::is_constructible_v<handler<T>, model,
                                                              std::function<Gtk::TreeRow()>>>>>
      : std::true_type {};

  template <class T>
  auto add(const std::string& name, const T& value, int interval = 1000)
      -> std::enable_if_t<has_handler<T>::value> {
    auto category       = *new_or_existing_row(handler<T>::category_name);
    auto base           = *store_->append(category.children());
    base[columns_.name] = name;

    auto h = handler<T>{columns_, [&] { return *store_->append(base.children()); }};
    Glib::signal_timeout().connect(
        [h, &value] {
          h.update(value);
          return true;
        },
        interval);
  }

private:
  Gtk::TreeModel::iterator new_or_existing_row(const std::string& name) {
    if (auto it = rows_.find(name); it != rows_.end()) {
      return it->second;
    }
    auto r = store_->append();
    rows_.insert({name, r});
    (*r)[columns_.name] = name;
    return r;
  }

  model columns_;
  std::unordered_map<std::string, Gtk::TreeModel::iterator> rows_;
  Glib::RefPtr<Gtk::TreeStore> store_;
};

template <class T>
struct status_tree::handler<T, std::void_t<decltype(std::declval<T>().total_messages()),
                                           decltype(std::declval<T>().messages_per_second()),
                                           decltype(std::declval<T>().parse_error()),
                                           decltype(std::declval<T>().last_updated())>> {
  static constexpr auto category_name = "Receivers";

  template <class F>
  handler(const status_tree::model& m, F add_row)
      : model{m}, r1{add_row()}, r2{add_row()}, r3{add_row()}, r4{add_row()} {
    r1[model.name] = "Total received messages";
    r2[model.name] = "Messages per second";
    r3[model.name] = "Parse error";
    r4[model.name] = "Last updated time";
  }

  void update(const T& receiver) const {
    const auto lu = receiver.last_updated();
    const auto tt = std::chrono::system_clock::to_time_t(lu);
    const auto e  = lu.time_since_epoch();
    const auto ms = (e - std::chrono::duration_cast<std::chrono::seconds>(e)) / 1ms;

    r1[model.value] = fmt::format("{}", receiver.total_messages());
    r2[model.value] = fmt::format("{}", receiver.messages_per_second());
    r3[model.value] = fmt::format("{}", receiver.parse_error());
    r4[model.value] = fmt::format("{:%T}.{:03d}", *std::localtime(&tt), ms);
  }

  const status_tree::model& model;
  Gtk::TreeRow r1, r2, r3, r4;
};

template <template <class> class T, class C>
struct status_tree::handler<T<C>, std::void_t<decltype(std::declval<T<C>>().connection()),
                                              decltype(std::declval<C>().total_messages()),
                                              decltype(std::declval<C>().messages_per_second()),
                                              decltype(std::declval<C>().total_errors()),
                                              decltype(std::declval<C>().last_sent())>> {
  static constexpr auto category_name = "Radio";

  template <class F>
  handler(const status_tree::model& m, F add_row)
      : model{m}, r1{add_row()}, r2{add_row()}, r3{add_row()}, r4{add_row()} {
    r1[model.name] = "Total sent messages";
    r2[model.name] = "Messages per second";
    r3[model.name] = "Total errors";
    r4[model.name] = "Last sent time";
  }

  void update(const T<C>& radio) const {
    const auto& c = radio.connection();
    const auto lu = c.last_sent();
    const auto tt = std::chrono::system_clock::to_time_t(lu);
    const auto e  = lu.time_since_epoch();
    const auto ms = (e - std::chrono::duration_cast<std::chrono::seconds>(e)) / 1ms;

    r1[model.value] = fmt::format("{}", c.total_messages());
    r2[model.value] = fmt::format("{}", c.messages_per_second());
    r3[model.value] = fmt::format("{}", c.total_errors());
    r4[model.value] = fmt::format("{:%T}.{:03d}", *std::localtime(&tt), ms);
  }

  const status_tree::model& model;
  Gtk::TreeRow r1, r2, r3, r4;
};

template <class T>
struct status_tree::handler<T, std::enable_if_t<std::is_same_v<
                                   decltype(std::declval<const T>().value()), model::refbox>>> {
  static constexpr auto category_name = "Updater";

  template <class F>
  handler(const status_tree::model& m, F add_row)
      : model{m},
        r1{add_row()},
        r2{add_row()},
        r3{add_row()},
        r4{add_row()},
        r5{add_row()},
        r6{add_row()} {
    r1[model.name] = "stage";
    r2[model.name] = "command";
    r3[model.name] = "abp_target";
    r4[model.name] = "goalie";
    r5[model.name] = "max allowed bots";
    r6[model.name] = "timeout";
  }

  void update(const T& updater) const {
    const auto v = updater.value();
    const auto bpp(v.ball_placement_position());

    r1[model.value] = stage_name_to_string(v.stage()).data();
    r2[model.value] = game_command_to_string(v.command()).data();
    r3[model.value] = fmt::format("{}, {}", std::floor(bpp.x()), std::floor(bpp.y()));
    r4[model.value] =
        fmt::format("Yel:{}, Blu:{}", v.team_yellow().goalie(), v.team_blue().goalie());
    r5[model.value] = fmt::format("Yel:{}, Blu:{}", v.team_yellow().max_allowed_bots(),
                                  v.team_blue().max_allowed_bots());
    r6[model.value] =
        fmt::format("Yel:{}|{}, Blu:{}|{}", v.team_yellow().timeouts(),
                    v.team_yellow().timeout_time() / 1000000, v.team_blue().timeouts(),
                    v.team_blue().timeout_time() / 1000000);
  }

  const status_tree::model& model;
  Gtk::TreeRow r1, r2, r3, r4, r5, r6;
};

auto main(int argc, char** argv) -> int {
  auto app = Gtk::Application::create(argc, argv);

  logger::sink::ostream sink(std::cout, "{elapsed} {level:<5} {zone}: {message}");

  log_area la{};
  logger::sink::function sink_log_area([&la](auto&& item) {
    Glib::signal_idle().connect_once([&la, item]() { la.write(item); });
  });

  logger::logger l{"main()"};
  l.info("(⋈◍＞◡＜◍)。✧♡");

#ifdef AI_SERVER_HAS_NNABLA_EXT_CUDA
  nbla::init_cudnn();
#endif
  l.info(fmt::format("nnabla: backend = {}, device_id = {}", nnabla_backend(),
                     nnabla_device_id()));

  try {
    // 設定ファイルのパスを決める
    // デフォルトは実行ファイルと同じディレクトリの config (<ai-server>/config への symlink)
    // argv[1] が指定されていたらそれを使う
    const auto config_dir =
        (argc == 2)
            ? std::filesystem::path{argv[1]}
            : weakly_canonical(std::filesystem::canonical(argv[0]).parent_path() / "config");
    l.info(fmt::format("configuration directory: {}", config_dir.c_str()));
    if (!is_directory(config_dir)) {
      l.error(fmt::format("'{}' does not exist or is not a directory", config_dir.c_str()));
      return -1;
    }

    // WorldModelの設定
    model::updater::world updater_world{};
    {
      // ロボットの速度と加速度を計算するか
      if (use_va_filter) {
        updater_world.robots_blue_updater()
            .set_default_filter<filter::va_calculator<model::robot>>();
        updater_world.robots_yellow_updater()
            .set_default_filter<filter::va_calculator<model::robot>>();
      }
      l.info("va filter (robot): "s + (use_va_filter ? "enabled"s : "disabled"s));
      // ボールの状態オブザーバを使うか
      if (use_ball_observer) {
        updater_world.ball_updater().set_filter<filter::state_observer::ball>(
            model::ball{}, std::chrono::system_clock::now());
      }
      l.info("state observer (ball): "s + (use_ball_observer ? "enabled"s : "disabled"s));
    }

    boost::asio::io_context receiver_io{1};

    // Vision receiverの設定
    std::atomic<bool> vision_received{false};
    receiver::vision vision{receiver_io, "0.0.0.0", vision_address, vision_port};
    vision.on_receive([&updater_world, &vision_received, &l](auto&& p) {
      if (!vision_received) {
        // 最初に受信したときにメッセージを表示する
        l.info("vision packet received!");
        vision_received = true;
      }
      updater_world.update(std::forward<decltype(p)>(p));
    });
    l.info(fmt::format("vision: {}:{}", vision_address, vision_port));

    // Refbox receiverの設定
    std::atomic<bool> refbox1_received{false};
    model::updater::refbox updater_refbox{};
    receiver::refbox refbox{receiver_io, "0.0.0.0", refbox_address, refbox_port};
    refbox.on_receive([&updater_refbox, &refbox1_received, &l](auto&& p) {
      if (!refbox1_received) {
        // 最初に受信したときにメッセージを表示する
        l.info("refbox packet received!");
        refbox1_received = true;
      }

      updater_refbox.update(std::forward<decltype(p)>(p));
    });
    l.info(fmt::format("refbox: {}:{}", refbox_address, refbox_port));

    // Robot receiverの設定
    std::atomic<bool> robot_received{false};
    receiver::robot robot{receiver_io, "0.0.0.0", robot_address, robot_port};
    robot.on_receive([&robot_received, &l](auto&& p) {
      if (!robot_received) {
        // 最初に受信したときにメッセージを表示する
        l.info("robot packet received!");
        robot_received = true;
      }
    });
    l.info(fmt::format("robot: {}:{}", robot_address, robot_port));

    // receiver_ioに登録されたタスクを別スレッドで開始
    std::thread receiver_thread{[&receiver_io, &l] {
      try {
        receiver_io.run();
      } catch (std::exception& e) {
        l.error(fmt::format("exception at receiver_thread: {}", e.what()));
      }
    }};
    util::set_thread_name(receiver_thread, "receiver_thread");
    stop_and_join_at_exit receiver_io_and_thread{receiver_io, std::move(receiver_thread)};

    boost::asio::io_context driver_io{1};

    // Radioの設定
    auto radio = [&] {
      if constexpr (is_grsim) {
        auto con = std::make_unique<radio::connection::udp>(
            driver_io, boost::asio::ip::udp::endpoint{
                           boost::asio::ip::make_address(grsim_address), grsim_command_port});
        l.info(fmt::format("radio: grSim ({}:{})", grsim_address, grsim_command_port));
        return std::make_shared<radio::grsim<radio::connection::udp>>(std::move(con));
      } else {
        if constexpr (use_udp) {
          auto con = std::make_unique<radio::connection::udp>(
              driver_io, boost::asio::ip::udp::endpoint{
                             boost::asio::ip::make_address(robot_address), robot_port});
          l.info(fmt::format("radio: kiks ({}:{})", robot_address, robot_port));
          return std::make_shared<radio::humanoid<radio::connection::udp>>(std::move(con));
        } else {
          auto con = std::make_unique<radio::connection::serial>(
              driver_io, xbee_path, radio::connection::serial::baud_rate(57600));
          l.info(fmt::format("radio: kiks ({})", xbee_path));
          return std::make_shared<radio::humanoid<radio::connection::serial>>(std::move(con));
        }
      }
    }();

    // driver による命令の送信を別スレッドで開始
    ai_server::driver driver{driver_io, cycle, updater_world, model::team_color::yellow};
    std::thread driver_thread{[&driver_io, &l] {
      try {
        driver_io.run();
      } catch (std::exception& e) {
        l.error(fmt::format("exception at driver_thread: {}", e.what()));
      }
    }};
    util::set_thread_name(driver_thread, "driver_thread");
    stop_and_join_at_exit driver_io_and_thread{driver_io, std::move(driver_thread)};

    refbox_panel rp{updater_refbox};

    game_runner runner{config_dir, updater_world, rp.updater(), driver, radio};
    game_panel gp{updater_world, runner};

    vision_panel vp{};
    vision_area va{updater_world};

    // vision_area と refbox_panel をつなげる
    va.signal_abp_target_changed().connect(sigc::mem_fun(rp, &refbox_panel::set_abp_target));

    // simulator の radio ならシグナル発火でボール, ロボット配置
    // generic lambda は template 相当のもので実現されるので constexpr if が使える
    [&va](auto& radio) {
      if constexpr (std::is_base_of_v<
                        radio::base::simulator,
                        typename std::remove_reference_t<decltype(radio)>::element_type>) {
        va.signal_ball_position_changed().connect(
            [radio](auto x, auto y) { radio->set_ball_position(x, y); });
        va.signal_robot_position_changed().connect(
            [radio](auto color, auto id, auto x, auto y, auto theta) {
              radio->set_robot_position(color, id, x, y, theta);
            });
      }
    }(radio);

    // スイッチによって vision_area の表示/非表示を切り替え
    vp.signal_switch_changed().connect(sigc::mem_fun(va, &vision_area::set_visible));

    gp.pack_start(vp, Gtk::PACK_SHRINK, 4);

    // refbox_panel を game_panel 内に追加する
    // TODO: game_panel を Frame 単位などに分割する？
    gp.pack_start(rp, Gtk::PACK_SHRINK, 4);

    Glib::signal_timeout().connect(
        [&vision_received, &gp, &l] {
          // Visionから値が取れてなかったら待つ
          if (!vision_received) return true;

          // 状態オブザーバなどの値が収束するまでもう少し待つ
          Glib::signal_timeout().connect_seconds_once(
              [&gp, &l] {
                l.info("ready!");
                gp.set_ready();
              },
              5);
          return false;
        },
        500);

    status_tree tree{};
    tree.add("Vision", vision);
    tree.add("RefBox", refbox);
    tree.add(is_grsim ? "grSim" : "KIKS", *radio);
    tree.add("Global Refbox", rp.updater().global(), 250);
    tree.add("Local Refbox", rp.updater().local(), 250);
    tree.expand_all();

    auto win      = Gtk::Window{};
    auto tree_win = Gtk::ScrolledWindow{};
    auto log_win  = Gtk::ScrolledWindow{};
    auto box0     = Gtk::HBox{};
    auto box1     = Gtk::VPaned{};
    auto box2     = Gtk::HBox{};
    {
      win.set_border_width(10);
      win.set_default_size(1280, -1);
      win.set_title("Game configurations");

      tree_win.add(tree);
      tree_win.set_min_content_width(350);

      box2.pack_end(tree_win, Gtk::PACK_SHRINK, 10);
      box2.pack_end(gp, Gtk::PACK_SHRINK, 10);

      log_win.add(la);
      la.signal_size_allocate().connect([&log_win]([[maybe_unused]] Gtk::Allocation& allloc) {
        auto adj = log_win.get_vadjustment();
        adj->set_value(adj->get_upper());
      });

      box1.set_wide_handle(true);
      box1.pack1(box2, Gtk::FILL);
      box1.pack2(log_win, Gtk::EXPAND);

      box0.pack_end(box1, Gtk::PACK_SHRINK, 0);
      box0.pack_end(va, Gtk::PACK_EXPAND_WIDGET, 10);

      win.add(box0);
      win.show_all_children();
    }
    app->run(win);
  } catch (std::exception& e) {
    l.error(e.what());
    return -1;
  } catch (...) {
    l.error("unknown error occurred");
    return -1;
  }
}
