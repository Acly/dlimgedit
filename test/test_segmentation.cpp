#include "image.hpp"
#include "segmentation.hpp"
#include "test_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <dlimgedit/dlimgedit.hpp>

#include <numeric>

namespace dlimgedit {

TEST_CASE("Image preparation", "[segmentation]") {
    auto shape = Shape{1, 3, 8, 10};
    auto img = Image(Extent{8, 6}, Channels::rgba);
    std::iota(img.pixels().begin(), img.pixels().end(), 0);
    auto tensor = create_image_tensor(img, shape);
    CHECK(tensor.dimensions()[0] == 1);
    CHECK(tensor.dimensions()[1] == 3);
    CHECK(tensor.dimensions()[2] == 8);
    CHECK(tensor.dimensions()[3] == 10);
    // red channel at index 2 (BGR), height x width
    CHECK(tensor(0, 2, 0, 0) == 0);
    CHECK(tensor(0, 2, 0, 1) == 4);
    CHECK(tensor(0, 2, 0, 2) == 8);
    CHECK(tensor(0, 2, 0, 3) == 12);
    CHECK(tensor(0, 2, 0, 4) == 16);
    CHECK(tensor(0, 2, 0, 5) == 20);
    CHECK(tensor(0, 2, 0, 6) == 24);
    CHECK(tensor(0, 2, 0, 7) == 28);
    CHECK(tensor(0, 2, 0, 8) == 0); // width padded up to 10
    CHECK(tensor(0, 2, 0, 9) == 0);
    CHECK(tensor(0, 2, 1, 0) == 32);
    CHECK(tensor(0, 2, 1, 1) == 36);
    CHECK(tensor(0, 2, 1, 2) == 40);
    //...
    CHECK(tensor(0, 2, 5, 7) == 188);
    CHECK(tensor(0, 2, 5, 8) == 0);
    CHECK(tensor(0, 2, 6, 7) == 0); // height padded up to 8
    CHECK(tensor(0, 2, 7, 7) == 0);
    // green channel
    CHECK(tensor(0, 1, 0, 0) == 1);
    CHECK(tensor(0, 1, 0, 1) == 5);
    CHECK(tensor(0, 1, 0, 2) == 9);
    // ...
    // blue channel at index 0
    CHECK(tensor(0, 0, 0, 0) == 2);
    CHECK(tensor(0, 0, 0, 1) == 6);
    CHECK(tensor(0, 0, 0, 2) == 10);
    // ...
}

TEST_CASE("Tensor to mask", "[segmentation]") {
    auto tensor_values = std::array{0.0f, 0.0f, 0.2f, -3.1f, 0.0f, 5.5f, 0.0f, 0.7f, 0.0f, 0.9f};
    auto tensor = TensorMap<float const, 4>(tensor_values.data(), Shape(1, 1, 2, 5));
    auto mask = create_mask_image(tensor, Extent{4, 2});
    auto result = as_tensor(mask);
    CHECK(result(0, 0, 0) == 0);
    CHECK(result(0, 1, 0) == 0);
    CHECK(result(0, 2, 0) == 255);
    CHECK(result(0, 3, 0) == 0);
    CHECK(result(1, 0, 0) == 255);
    CHECK(result(1, 1, 0) == 0);
    CHECK(result(1, 2, 0) == 255);
    CHECK(result(1, 3, 0) == 0);
}

TEST_CASE("Segmentation", "[segmentation]") {
    auto env = default_env();
    auto img = Image::load((test_dir() / "input" / "cat_and_hat.png").string());
    auto seg = Segmentation::process(img, env);

    SECTION("point") {
        auto mask = seg.get_mask(Point{320, 210});
        check_image_matches(mask, "test_segmentation_point.png");
    }
    SECTION("region") {
        auto mask = seg.get_mask(Region{Point{180, 110}, Extent{325, 220}});
        check_image_matches(mask, "test_segmentation_region.png");
    }
}

TEST_CASE("Segmentation on GPU", "[segmentation]") {
    auto model_path = model_dir().string();
    auto opts = Options{};
    opts.device = Device::gpu;
    opts.model_path = model_path;
    auto env = Environment(opts);
    auto img = Image::load((test_dir() / "input" / "cat_and_hat.png").string());
    auto seg = Segmentation::process(img, env);
    auto mask = seg.get_mask(Point{320, 210});
    check_image_matches(mask, "test_segmentation_gpu.png");
}

} // namespace dlimgedit