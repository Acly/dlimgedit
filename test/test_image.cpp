#include "test_utils.hpp"
#include <dlimgedit/dlimgedit.hpp>
#include <image.hpp>

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

TEST_CASE("Image load", "[image]") {
    auto const img = Image::load(test_dir() / "input" / "cat_and_hat.png");
    CHECK(img.extent().width == 512);
    CHECK(img.extent().height == 512);
    CHECK(img.channels() == Channels::rgba);
    CHECK(img.size() == 512 * 512 * 4);
}

TEST_CASE("Image save", "[image]") {
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
        CHECK(result.pixels()[i * 4 + 0] == 255);
        CHECK(result.pixels()[i * 4 + 1] == i);
        CHECK(result.pixels()[i * 4 + 2] == 0);
        CHECK(result.pixels()[i * 4 + 3] == 255);
    }
}

TEST_CASE("Image resize", "[image]") {
    auto img = Image(Extent(8, 8), Channels::rgba);
    for (int i = 0; i < 8 * 8; ++i) {
        img.pixels()[i * 4 + 0] = 255;
        img.pixels()[i * 4 + 1] = 4 * (i / 8);
        img.pixels()[i * 4 + 2] = 4 * (i % 8);
        img.pixels()[i * 4 + 3] = 255;
    }
    auto const result = resize(img, Extent{4, 4});
    REQUIRE(result.extent().width == 4);
    REQUIRE(result.extent().height == 4);
    REQUIRE(result.channels() == Channels::rgba);
    for (int i = 0; i < 16; ++i) {
        CHECK(result.pixels()[i * 4 + 0] == 255);
        CHECK(int(result.pixels()[i * 4 + 1]) == 2 + 8 * (i / 4));
        CHECK(int(result.pixels()[i * 4 + 2]) == 2 + 8 * (i % 4));
        CHECK(result.pixels()[i * 4 + 3] == 255);
    }
}

} // namespace dlimgedit
