"""Exports the MobileSAM model to ONNX format."""

import sys
import os
import argparse


def run(mobile_sam_root, mobile_sam_onnx_root, output_dir):
    sys.path.append(mobile_sam_root)
    sys.path.append(mobile_sam_onnx_root)
    from scripts import export_onnx_model
    from mobile_sam_encoder_onnx import export_image_encoder

    os.makedirs(output_dir, exist_ok=True)

    input_weights = mobile_sam_root + "/weights/mobile_sam.pt"
    output_encoder = output_dir + "mobile_sam_image_encoder.onnx"
    output_decoder_single = output_dir + "sam_mask_decoder_single.onnx"
    output_decoder_multi = output_dir + "sam_mask_decoder_multi.onnx"

    export_image_encoder.run_export(
        model_type="vit_t",
        checkpoint=input_weights,
        output=output_encoder,
        opset=17,
        use_preprocess=True,
    )

    export_onnx_model.run_export(
        model_type="vit_t",
        checkpoint=input_weights,
        output=output_decoder_single,
        opset=17,
        return_single_mask=True,
    )

    export_onnx_model.run_export(
        model_type="vit_t",
        checkpoint=input_weights,
        output=output_decoder_multi,
        opset=17,
        return_single_mask=False,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--mobile-sam",
        type=str,
        default="MobileSAM",
        help="Path to MobileSAM root directory (https://github.com/ChaoningZhang/MobileSAM)",
    )
    args.add_argument(
        "--mobile-sam-onnx",
        type=str,
        default="MobileSAM-onnx",
        help="Path to MobileSAM-onnx root directory (https://huggingface.co/Acly/MobileSAM)",
    )
    args.add_argument(
        "--output-dir",
        type=str,
        default=".onnx/",
        help="Output directory for ONNX models",
    )
    args = args.parse_args()
    run(args.mobile_sam, args.mobile_sam_onnx, args.output_dir)
