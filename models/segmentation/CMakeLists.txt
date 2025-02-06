message(STATUS "Downloading segmentation models: MobileSAM")
file(DOWNLOAD
    https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/mobile_sam_image_encoder.onnx
    EXPECTED_HASH MD5=9E0ED7F27DC33C6DFD08A0CBA6EAC141
)
file(DOWNLOAD
    https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_multi.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/sam_mask_decoder_multi.onnx
    EXPECTED_HASH MD5=CFF1C936628337B5F4D4EFAD9F94CCA7
)
file(DOWNLOAD
    https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/sam_mask_decoder_single.onnx
    EXPECTED_HASH MD5=5A5174CCF1A62EC4FFF38E2ACBBD8201
)

message(STATUS "Downloading segmentation models: BiRefNet")
file(DOWNLOAD
    https://huggingface.co/onnx-community/BiRefNet_lite/resolve/main/onnx/model.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/birefnet_cpu.onnx
    EXPECTED_HASH SHA256=5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333
)
file(DOWNLOAD
    https://huggingface.co/onnx-community/BiRefNet_lite/resolve/main/onnx/model_fp16.onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/birefnet_gpu.onnx
    EXPECTED_HASH SHA256=d39b897ceb16ae654c1731f3dba0cf9b368d9cae74b5a57459b455cc8bfec402
)
# file(DOWNLOAD
#     https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet_lite-general-2K-epoch_232.onnx
#     ${CMAKE_CURRENT_SOURCE_DIR}/birefnet_hr_cpu.onnx
#     EXPECTED_HASH SHA256=6003D2F758BDB4E4802A09E39167529BC2EEF9288D5B8FA537331467CBC4759D
# )