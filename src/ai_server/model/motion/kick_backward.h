#ifndef AI_SERVER_MODEL_MOTION_KICK_BACK_H
#define AI_SERVER_MODEL_MOTION_KICK_BACK_H

#include "base.h"

namespace ai_server::model::motion {

    class kick_backward : public base {

        public:
            kick_backward();

            std::tuple<double, double, double> execute() override;
    };
}

#endif