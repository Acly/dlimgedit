#ifndef DLIMGEDIT_H_
#define DLIMGEDIT_H_

#ifdef DLIMGEDIT_EXPORTS
#    ifdef _MSC_VER
#        define DLIMG_API __declspec(dllexport)
#    else
#        define DLIMG_API __attribute__((visibility("default")))
#    endif
#else
#    ifdef _MSC_VER
#        define DLIMG_API __declspec(dllimport)
#    else
#        define DLIMG_API
#    endif
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dlimg_Environment_* dlimg_Environment;
typedef struct dlimg_Segmentation_* dlimg_Segmentation;

typedef struct dlimg_ImageView {
    int width;
    int height;
    int channels;
    int stride;
    uint8_t* pixels;
} dlimg_ImageView;

typedef enum dlimg_Backend { dlimg_cpu, dlimg_gpu } dlimg_Backend;

typedef struct dlimg_Options {
    dlimg_Backend backend;
    char const* model_path;
} dlimg_Options;

typedef enum dlimg_Result { dlimg_success, dlimg_error } dlimg_Result;

struct dlimg_Api {
    int (*is_backend_supported)(dlimg_Backend);

    dlimg_Result (*create_environment)(dlimg_Environment*, dlimg_Options const*);
    void (*destroy_environment)(dlimg_Environment);

    dlimg_Result (*process_image_for_segmentation)(dlimg_Segmentation*, dlimg_ImageView const*,
                                                   dlimg_Environment);
    dlimg_Result (*get_segmentation_mask)(dlimg_Segmentation, int const* /*point*/,
                                          int const* /*region*/, uint8_t** /*out_masks*/,
                                          float* /*out_accuracys*/);
    void (*get_segmentation_extent)(dlimg_Segmentation, int* /*out_extent*/);
    void (*destroy_segmentation)(dlimg_Segmentation);

    dlimg_Result (*load_image)(char const*, int* /*out_extent*/, int* /*out_channels*/,
                               uint8_t** /*out_pixels*/);
    dlimg_Result (*save_image)(dlimg_ImageView const*, char const*);
    uint8_t* (*create_image)(int, int, int);
    void (*destroy_image)(uint8_t const*);

    char const* (*last_error)();
};

DLIMG_API dlimg_Api const* dlimg_init();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // DLIMGEDIT_H_
