#include "image.hpp"
#include "segmentation.hpp"
#include "test_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <dlimgedit/dlimgedit.hpp>

#include <numeric>

namespace dlimgedit {

TEST_CASE("Image preparation", "[segmentation]") {
    auto shape = std::array<int64_t, 4>{1, 3, 10, 8};
    auto img = Image(Extent{8, 6}, Channels::rgba);
    std::iota(img.pixels().begin(), img.pixels().end(), 0);
    auto tensor = create_image_tensor(img, shape);
    CHECK(tensor.size() == 1 * 3 * 10 * 8);
    // red channel, height x width
    CHECK(tensor[0] == 0);
    CHECK(tensor[1] == 32);
    CHECK(tensor[2] == 64);
    CHECK(tensor[3] == 96);
    CHECK(tensor[4] == 128);
    CHECK(tensor[5] == 160);
    CHECK(tensor[6] == 0); // padded up to 10
    CHECK(tensor[7] == 0);
    CHECK(tensor[8] == 0);
    CHECK(tensor[9] == 0);
    CHECK(tensor[10] == 4);
    CHECK(tensor[11] == 36);
    CHECK(tensor[12] == 68);
    //...
    // green channel
    CHECK(tensor[80] == 1);
    CHECK(tensor[81] == 33);
    CHECK(tensor[82] == 65);
    // ...
    // blue channel
    CHECK(tensor[160] == 2);
    CHECK(tensor[161] == 34);
    CHECK(tensor[162] == 66);
    // ...
}

TEST_CASE("Tensor to mask", "[segmentation]") {
    auto tensor_values = std::array{0.0f, 0.0f, 0.2f, 0.3f, 0.0f, 0.5f, 0.0f, 0.7f, 0.0f, 0.9f};
    auto tensor = ImageAccess<const float>(Extent{2, 5}, 1, tensor_values.data());
    auto mask = create_mask_image(tensor, Extent{4, 2});
    auto maskv = ImageAccess<const uint8_t>(mask);
    CHECK(maskv(0, 0) == 0);
    CHECK(maskv(0, 1) == 0);
    CHECK(maskv(1, 0) == 255);
    CHECK(maskv(1, 1) == 255);
    CHECK(maskv(2, 0) == 0);
    CHECK(maskv(2, 1) == 255);
    CHECK(maskv(3, 0) == 0);
    CHECK(maskv(3, 1) == 255);
}

TEST_CASE("Segmentation from point", "[segmentation]") {
    auto env = default_env();
    auto img = Image::load((test_dir() / "input" / "cat_and_hat.png").string());
    auto seg = Segmentation::process(img, env);
    auto mask = seg.get_mask(Point{220, 310});
    CHECK(mask.extent().width == 512);
    CHECK(mask.extent().height == 512);
    CHECK(mask.channels() == Channels::mask);
    Image::save(mask, (test_dir() / "result" / "test_segmentation_from_point.png").string());
}

} // namespace dlimgedit