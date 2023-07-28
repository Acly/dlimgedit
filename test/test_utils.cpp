#include "test_utils.hpp"
#include "tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <cmath>

namespace dlimg {

class ResultCleanup : public Catch::EventListenerBase {
  public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(Catch::TestRunInfo const&) override {
        auto results_dir = test_dir() / "result";
        if (exists(results_dir)) {
            remove_all(results_dir);
        }
        create_directory(results_dir);
    }
};

CATCH_REGISTER_LISTENER(ResultCleanup)

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
    auto model_directory = model_dir().string();
    Options opts;
    opts.backend = Backend::cpu;
    opts.model_directory = model_directory.c_str();
    return Environment(opts);
}

float rmse(ImageView const& image_a, ImageView const& image_b) {
    REQUIRE(image_a.extent == image_b.extent);
    REQUIRE(image_a.channels == image_b.channels);
    auto a = as_tensor(image_a).cast<float>() / 255.f;
    auto b = as_tensor(image_b).cast<float>() / 255.f;
    auto squared_diff = (a - b).square();
    Tensor<float, 0> mean = squared_diff.mean();
    return std::sqrt(*mean.data());
}

void check_image_matches(ImageView const& image, std::string_view reference, float threshold) {
    auto img_path = test_dir() / "result" / reference;
    Image::save(image, img_path);

    auto ref_path = test_dir() / "reference" / reference;
    auto ref_image = Image::load(ref_path);
    auto error = rmse(image, ref_image);
    INFO("Comparing " << img_path << " to " << ref_path << ": RMSE=" << error);
    CHECK(error < threshold);
}

} // namespace dlimg
