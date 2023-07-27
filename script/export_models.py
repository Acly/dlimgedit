"""Exports the MobileSAM model to ONNX format."""

import sys
import os


def run(mobile_sam_root, output_dir):
    sys.path.append(mobile_sam_root)
    from scripts import export_image_encoder, export_onnx_model

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
    sam_root = sys.argv[1] if len(sys.argv) > 1 else "mobile_sam"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else ".onnx/"
    run(sam_root, output_dir)
