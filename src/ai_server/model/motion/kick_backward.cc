#include kick_backward.h

namespace ai_server::model::motion {

    kick_backward::kick_backward() : base(31) {}

    std::tuple<double, double, double> kick_backward::execute() {
        return std::make_tuple<double, double, double>(100.0, 0.0, 0.0);
    }
}