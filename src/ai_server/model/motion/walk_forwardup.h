#ifndef AI_SERVER_MODEL_MOTION_WALK_FORWARDUP_H
#define AI_SERVER_MODEL_MOTION_WALK_FORWARDUP_H

#include "base.h"

namespace ai_server::model::motion {

class walk_forwardup : public base {
public:
  walk_forwardup();

  std::tuple<double, double, double> execute() override;
};

} // namespace ai_server::model::motion

#endif
