#Int8 Quantizer 
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8)

    print(f"quantized model saved to:{quantized_model_path}")

quantize_onnx_model("/home/vimal/Project/models/yolop-640-640.onnx", "model_quant.onnx")