message(STATUS "Downloading segmentation models")
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
