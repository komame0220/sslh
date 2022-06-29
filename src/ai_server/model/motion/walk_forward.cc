/*
#include "walk_forward.h"

namespace ai_server::model::motion {

walk_forward::walk_forward() : base(1) {}

std::tuple<double, double, double> walk_forward::execute() {
  return std::make_tuple<double, double, double>(100.0, 0.0, 0.0);
}

} // namespace ai_server::model::motion
*/

#include "walk_forward.h"

namespace ai_server::model::motion {

  walk_forward::walk_forward() : base(16) {}

  std::tuple<double, double, double> walk_forward::execute() {
    return std::make_tuple<double, double, double>(100.0, 0.0, 0.0);
  }
}