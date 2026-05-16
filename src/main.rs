mod gen_cpp;
mod operators;
mod utils;

use color_eyre::eyre::Result;
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    color_eyre::install()?;
    let model_path = "/tmp/train_step/model.onnx";
    let Ok(model_proto) = tract_onnx::onnx().proto_model_for_path(model_path) else {
        return Err(color_eyre::eyre::eyre!("Failed to load model proto"));
    };
    let Ok(_model) = tract_onnx::onnx().model_for_proto_model(&model_proto) else {
        return Err(color_eyre::eyre::eyre!("Failed to load model from proto"));
    };
    let cpp_code = gen_cpp::generate_cpp_code(model_proto)?;
    println!("{}", cpp_code);
    Ok(())
}
