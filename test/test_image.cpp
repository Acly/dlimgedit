#include "test_utils.hpp"
#include <dlimgedit/dlimgedit.hpp>
#include <image.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace dlimg {

TEST_CASE("Image formats", "[image]") {
    auto channels =
        GENERATE(Channels::mask, Channels::rgb, Channels::rgba, Channels::bgra, Channels::argb);
    auto const img = Image(Extent{8, 6}, channels);
    CHECK(img.size() == 8 * 6 * count(channels));
    CHECK(img.size() <= 8 * 6 * 4);
    CHECK(img.size() >= 8 * 6 * 1);
}

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
        img.pixels()[i * 4 + 0] = uint8_t(255);
        img.pixels()[i * 4 + 1] = uint8_t(i);
        img.pixels()[i * 4 + 2] = uint8_t(0);
        img.pixels()[i * 4 + 3] = uint8_t(255);
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
        img.pixels()[i * 4 + 0] = uint8_t(255);
        img.pixels()[i * 4 + 1] = uint8_t(4 * (i / 8));
        img.pixels()[i * 4 + 2] = uint8_t(4 * (i % 8));
        img.pixels()[i * 4 + 3] = uint8_t(255);
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

} // namespace dlimg
