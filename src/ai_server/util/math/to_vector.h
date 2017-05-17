#ifndef AI_SERVER_UTIL_MATH_TO_VECTOR_H
#define AI_SERVER_UTIL_MATH_TO_VECTOR_H

#include <type_traits>

#include <Eigen/Dense>

#include "ai_server/model/command.h"
#include "ai_server/util/math/detail/is_vector_type.h"

namespace ai_server {
namespace util {
namespace math {

/// @brief     メンバ関数x(), y()を持つオブジェクトを2次元のベクトル型に変換する
/// @param obj 変換したいオブジェクト
/// @return    Eigen::Matrix<T, 2, 1>{obj.x(), obj.y()}
template <class T, std::enable_if_t<detail::has_x_y_v<T>>* = nullptr>
inline auto position(const T& obj)
    -> Eigen::Matrix<std::common_type_t<decltype(obj.x()), decltype(obj.y())>, 2, 1> {
  return {obj.x(), obj.y()};
}

/// @brief     メンバ関数vx(), vy()を持つオブジェクトを2次元のベクトル型に変換する
/// @param obj 変換したいオブジェクト
/// @return    Eigen::Matrix<T, 2, 1>{obj.vx(), obj.vy()}
template <class T, std::enable_if_t<detail::has_vx_vy_v<T>>* = nullptr>
inline auto velocity(const T& obj)
    -> Eigen::Matrix<std::common_type_t<decltype(obj.vx()), decltype(obj.vy())>, 2, 1> {
  return {obj.vx(), obj.vy()};
}

/// @brief     メンバ関数ax(), ay()を持つオブジェクトを2次元のベクトル型に変換する
/// @param obj 変換したいオブジェクト
/// @return    Eigen::Matrix<T, 2, 1>{obj.ax(), obj.ay()}
template <class T, std::enable_if_t<detail::has_ax_ay_v<T>>* = nullptr>
inline auto acceleration(const T& obj)
    -> Eigen::Matrix<std::common_type_t<decltype(obj.ax()), decltype(obj.ay())>, 2, 1> {
  return {obj.ax(), obj.ay()};
}

} // namespace math
} // namespace util
} // namespace ai_server

#endif // AI_SERVER_UTIL_MATH_TO_VECTOR_H
