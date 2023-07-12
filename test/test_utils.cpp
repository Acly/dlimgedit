#include "test_utils.hpp"

namespace dlimgedit {

Path const& root_dir() {
    static Path const path = []() {
        Path cur = std::filesystem::current_path();
        while (!exists(cur / "README.md")) {
            cur = cur.parent_path();
            if (cur.empty() || cur == cur.parent_path()) {
                throw std::runtime_error("root directory not found");
            }
        }
        return cur;
    }();
    return path;
}

Path const& model_dir() {
    static const Path path = root_dir() / "models";
    return path;
}

Path const& test_dir() {
    static const Path path = root_dir() / "test";
    return path;
}

Environment default_env() {
    auto model_path = model_dir().string();
    Options opts;
    opts.device = Device::cpu;
    opts.model_path = model_path;
    return Environment(opts);
}

} // namespace dlimgedit
