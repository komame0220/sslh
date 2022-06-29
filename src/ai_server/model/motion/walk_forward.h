/*
#ifndef AI_SERVER_MODEL_MOTION_WALK_FORWARD_H
#define AI_SERVER_MODEL_MOTION_WALK_FORWARD_H

#include "base.h"

namespace ai_server::model::motion {

class walk_forward : public base {
public:
  walk_forward();

  std::tuple<double, double, double> execute() override;
};

} // namespace ai_server::model::motion

#endif
*/

#ifndef AI_SERVER_MODEL_MOTION_WALK_FORWARD_H
#define AI_SERVER_MODEL_MOTION_WALK_FORWARD_H

#include "base.h"

namespace ai_server::model::motion {

  class walk_forward : public base {
  public:
    walk_forward();

    std::tuple<double, double, double> execute() override;
  };
}

#endif