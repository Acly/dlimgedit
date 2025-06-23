#include "test_utils.hpp"
#include <dlimgedit/dlimgedit.hpp>
#include <image.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace dlimg {

TEST_CASE("Image formats", "[image]") {
    auto channels = GENERATE(
        Channels::mask, Channels::rgb, Channels::rgba, Channels::bgra, Channels::argb);
    auto const img = Image(Extent{8, 6}, channels);
    CHECK(img.size() == size_t(8 * 6 * count(channels)));
    CHECK(img.size() <= size_t(8 * 6 * 4));
    CHECK(img.size() >= size_t(8 * 6 * 1));
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

TEST_CASE("Tile merge", "[image]") {
    std::array<std::array<rgb32_t, 5 * 5>, 4> tiles;
    for (int t = 0; t < 4; ++t) {
        float v = float(t);
        std::fill(tiles[t].begin(), tiles[t].end(), rgb32_t{v, v, v});
    }
    std::array<rgb32_t, 8 * 8> dst{};
    auto dst_span = image_span<rgb32_t>(Extent(8, 8), dst.data());
    auto const layout = tile_layout(Extent{8, 8}, 6, 2, 1);
    merge_tile({Extent(5, 5), tiles[0].data()}, dst_span, {0, 0}, layout);
    merge_tile({Extent(5, 5), tiles[1].data()}, dst_span, {1, 0}, layout);
    merge_tile({Extent(5, 5), tiles[2].data()}, dst_span, {0, 1}, layout);
    merge_tile({Extent(5, 5), tiles[3].data()}, dst_span, {1, 1}, layout);

    float e00 = float(4 * 0 + 2 * 1 + 2 * 2 + 1 * 3) / 9.f;
    float e10 = float(2 * 0 + 4 * 1 + 1 * 2 + 2 * 3) / 9.f;
    float e01 = float(2 * 0 + 1 * 1 + 4 * 2 + 2 * 3) / 9.f;
    float e11 = float(1 * 0 + 2 * 1 + 2 * 2 + 4 * 3) / 9.f;
    // clang-format off
    auto expected_float = std::array<float, 8 * 8>{
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        2.f/3.f, 2.f/3.f, 2.f/3.f, e00    , e10    , 5.f/3.f, 5.f/3.f, 5.f/3.f,
        4.f/3.f, 4.f/3.f, 4.f/3.f, e01    , e11    , 7.f/3.f, 7.f/3.f, 7.f/3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f
    };
    // clang-format on
    std::array<rgb32_t, 8 * 8> expected;
    for (int i = 0; i < 8 * 8; ++i) {
        expected[i] = rgb32_t{expected_float[i], expected_float[i], expected_float[i]};
    }
    for (int i = 0; i < int(dst.size()); ++i) {
        CHECK(std::abs(dst[i][0] - expected[i][0]) < 0.0001f);
        CHECK(std::abs(dst[i][1] - expected[i][1]) < 0.0001f);
        CHECK(std::abs(dst[i][2] - expected[i][2]) < 0.0001f);
    }
}

TEST_CASE("Tile merge blending", "[image]") {
    std::array<rgb32_t, 22 * 19> dst{};
    auto dst_span = image_span<rgb32_t>(Extent(22, 19), dst.data());

    auto layout = tile_layout(Extent{22, 19}, 10, 3, 2);
    auto te = Extent(layout.tile_size[0], layout.tile_size[1]);
    auto tile = std::vector<rgb32_t>(te.width * te.height, rgb32_t{1.f, 1.f, 1.f});
    auto tile_span = image_span<rgb32_t>(te, tile.data());

    for (int y = 0; y < layout.n_tiles[1]; ++y) {
        for (int x = 0; x < layout.n_tiles[0]; ++x) {
            merge_tile(tile_span, dst_span, {x, y}, layout);
        }
    }
    for (int y = 0; y < dst_span.extent.height; ++y) {
        for (int x = 0; x < dst_span.extent.width; ++x) {
            CHECK(dst_span.get(x, y) == float4{1.f, 1.f, 1.f, 1.f});
        }
    }
}

} // namespace dlimg
