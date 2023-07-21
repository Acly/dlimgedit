#include "image.hpp"
#include "segmentation.hpp"
#include "test_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <dlimgedit/dlimgedit.hpp>

#include <format>
#include <numeric>

namespace dlimgedit {

TEST_CASE("Resize longest side", "[segmentation]") {
    SECTION("height") {
        auto img = Image(Extent{13, 19}, Channels::rgba);
        SECTION("upscale") {
            auto result = ResizeLongestSide(26);
            auto resized = result.resize(img);
            CHECK(resized.extent.height == 26);
            CHECK(resized.extent.width == 18);
        }
        SECTION("downscale") {
            auto result = ResizeLongestSide(10);
            auto resized = result.resize(img);
            CHECK(resized.extent.height == 10);
            CHECK(resized.extent.width == 7);
        }
    }
    SECTION("width") {
        auto img = Image(Extent{19, 13}, Channels::rgba);
        SECTION("upscale") {
            auto result = ResizeLongestSide(26);
            auto resized = result.resize(img);
            CHECK(resized.extent.height == 18);
            CHECK(resized.extent.width == 26);
        }
        SECTION("downscale") {
            auto result = ResizeLongestSide(10);
            auto resized = result.resize(img);
            CHECK(resized.extent.height == 7);
            CHECK(resized.extent.width == 10);
        }
    }
}

TEST_CASE("Resize and transform", "[segmentation]") {
    auto img = Image(Extent{10, 10}, Channels::rgba);
    auto img_size = ResizeLongestSide(20);
    auto resized = img_size.resize(img);
    CHECK(resized.extent.height == 20);
    CHECK(resized.extent.width == 20);
    CHECK(img_size.transform(Point{0, 0}) == Point{0, 0});
    CHECK(img_size.transform(Point{10, 10}) == Point{20, 20});
    CHECK(img_size.transform(Point{2, 7}) == Point{4, 14});
}

TEST_CASE("Image to tensor", "[segmentation]") {
    auto img = Image(Extent{8, 6}, Channels::rgba);
    std::iota(img.pixels(), img.pixels() + img.size(), 0);
    auto tensor = create_image_tensor(img);
    CHECK(tensor(0, 0, 0) == 0.f);
    CHECK(tensor(0, 0, 1) == 1.f);
    CHECK(tensor(0, 1, 0) == 4.f);
    CHECK(tensor(0, 1, 1) == 5.f);
    CHECK(tensor(1, 0, 0) == 32.f);
}

TEST_CASE("Tensor to mask", "[segmentation]") {
    auto tensor_values = std::array{0.0f, 0.0f, 0.2f, -3.1f, 0.0f, 5.5f, 0.0f, 0.7f, 0.0f, 0.9f};
    auto tensor = TensorMap<float const, 4>(tensor_values.data(), Shape(1, 1, 2, 5));
    auto mask = Image(Extent{4, 2}, Channels::mask);
    write_mask_image(tensor, 0, Extent{4, 2}, mask.pixels());
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
    auto img = Image::load(test_dir() / "input" / "cat_and_hat.png");
    auto seg = Segmentation::process(img, env);

    SECTION("point") {
        auto mask = seg.get_mask(Point{320, 210});
        check_image_matches(mask, "test_segmentation_point.png");
    }
    SECTION("region") {
        auto mask = seg.get_mask(Region{Point{180, 110}, Extent{325, 220}});
        check_image_matches(mask, "test_segmentation_region.png");
    }
    SECTION("all masks") {
        auto masks = seg.get_masks(Point{320, 210});
        CHECK(masks[0].accuracy >= 0.95f);
        CHECK(masks[1].accuracy >= 0.95f);
        CHECK(masks[2].accuracy >= 0.95f);
        check_image_matches(masks[0].image, "test_segmentation_point_0.png");
        check_image_matches(masks[1].image, "test_segmentation_point_1.png");
        check_image_matches(masks[2].image, "test_segmentation_point_2.png");
    }
}

TEST_CASE("Segmentation on GPU", "[segmentation]") {
    auto model_path = model_dir().string();
    auto opts = Options{};
    opts.device = Device::gpu;
    opts.model_path = model_path.c_str();
    auto env = Environment(opts);
    {
        auto img = Image::load(test_dir() / "input" / "cat_and_hat.png");
        auto seg = Segmentation::process(img, env);
        auto mask = seg.get_mask(Point{320, 210});
        check_image_matches(mask, "test_segmentation_gpu_hat.png");
    }
    {
        auto img = Image::load(test_dir() / "input" / "truck.jpg");
        auto seg = Segmentation::process(img, env);
        auto mask = seg.get_mask(Point{486, 722});
        check_image_matches(mask, "test_segmentation_gpu_truck.png");
    }
    {
        auto img = Image::load(test_dir() / "input" / "wardrobe.png");
        auto seg = Segmentation::process(img, env);
        auto mask1 = seg.get_mask(Point{136, 211});
        check_image_matches(mask1, "test_segmentation_gpu_wardrobe_1.png");
        auto mask2 = seg.get_mask(Point{45, 450});
        check_image_matches(mask2, "test_segmentation_gpu_wardrobe_2.png");
    }
}

} // namespace dlimgedit