#include "test_utils.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace dlimgedit {

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

TEST_CASE("Image can be loaded from file", "[image]") {
    auto const img = Image::load(test_dir() / "input" / "cat_and_hat.png");
    REQUIRE(img.extent().width == 512);
    REQUIRE(img.extent().height == 512);
    REQUIRE(img.channels() == Channels::rgba);
    REQUIRE(img.size() == 512 * 512 * 4);
}

TEST_CASE("Image can be saved to file", "[image]") {
    auto img = Image(Extent{16, 16}, Channels::rgba);
    for (int i = 0; i < 16 * 16; ++i) {
        img.pixels()[i * 4 + 0] = 255;
        img.pixels()[i * 4 + 1] = i;
        img.pixels()[i * 4 + 2] = 0;
        img.pixels()[i * 4 + 3] = 255;
    }
    auto filepath = test_dir() / "result" / "test_image_save.png";
    Image::save(img, filepath);
    REQUIRE(exists(filepath));

    auto const result = Image::load(filepath);
    REQUIRE(result.extent().width == 16);
    REQUIRE(result.extent().height == 16);
    REQUIRE(result.channels() == Channels::rgba);
    for (int i = 0; i < 16 * 16; ++i) {
        REQUIRE(result.pixels()[i * 4 + 0] == 255);
        REQUIRE(result.pixels()[i * 4 + 1] == i);
        REQUIRE(result.pixels()[i * 4 + 2] == 0);
        REQUIRE(result.pixels()[i * 4 + 3] == 255);
    }
}

} // namespace dlimgedit
