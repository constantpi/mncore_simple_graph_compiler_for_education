use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{ModelProto, ValueInfoProto, tensor_shape_proto::dimension, type_proto};
use tract_onnx::prelude::*;

fn value_info_to_type_vector(value_info: &ValueInfoProto) -> Option<(Vec<usize>, i32)> {
    let Some(type_proto) = &value_info.r#type else {
        return None;
    };
    let Some(tensor_type) = &type_proto.value else {
        return None;
    };
    let type_proto::Value::TensorType(tensor) = tensor_type else {
        return None;
    };
    let elem_type = tensor.elem_type;
    let Some(shape) = &tensor.shape else {
        return None;
    };
    let Some(dim) = shape
        .dim
        .iter()
        .map(|d| d.value.clone())
        .collect::<Option<Vec<_>>>()
    else {
        return None;
    };
    let Some(dim) = dim
        .iter()
        .map(|d| {
            if let dimension::Value::DimValue(size) = d {
                Some(*size as usize)
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()
    else {
        return None;
    };
    Some((dim, elem_type))
}

pub fn generate_cpp_code(model: ModelProto) -> Result<String> {
    let graph = model
        .graph
        .ok_or_else(|| color_eyre::eyre::eyre!("Model proto does not contain a graph"))?;
    let mut lines = vec![
        "#include \"matrix_operations.hpp\"".to_string(),
        "".to_string(),
        "// ctypes用のエクスポート関数".to_string(),
        "extern \"C\" {".to_string(),
        "".to_string(),
        "// OUTPUT_INFO_START".to_string(),
    ];

    // 出力の形状をコメントとして追加
    for out in graph.output.iter() {
        let name = out.name.clone();
        let Some((dim, _elem_type)) = value_info_to_type_vector(&out) else {
            continue;
        };
        if dim.is_empty() || (dim.len() == 1 && dim[0] == 1 && name.contains("loss")) {
            // スカラーの場合は shape=[] として表現
            lines.push(format!("// OUTPUT: {} shape=[]", out.name));
        } else {
            lines.push(format!(
                "// OUTPUT: {} shape=[{}]",
                out.name,
                dim.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    lines.push("// OUTPUT_INFO_END".to_string());
    lines.push("".to_string());

    let mut sig_lines = vec!["void forward_backward(".to_string()];
    // ONNXからすべての入力名を取得
    let mut param_names = vec![];
    for input in graph.input.iter() {
        let name = input.name.clone();
        param_names.push(name.clone());
        let Some((_dim, elem_type)) = value_info_to_type_vector(&input) else {
            continue;
        };
        if elem_type == 6 || elem_type == 7 {
            //INT32, INT64 は int* として表現
            sig_lines.push(format!("    const int* {name}_ptr,"));
        } else {
            // その他の型は float* として表現
            sig_lines.push(format!("    const float* {name}_ptr,"));
        }
    }

    // ONNXグラフ出力に基づく出力ポインタ
    // 出力名の重複を処理するためのカウンタ
    let mut output_counts = HashMap::new();
    for output in graph.output.iter() {
        let cnt = output_counts.entry(output.name.clone()).or_insert(0);
        let unique_name = if *cnt == 0 {
            output.name.clone()
        } else {
            format!("{}_{}", output.name, cnt)
        };
        *cnt += 1;

        // 出力の型に基づいてポインタの型を決定
        let Some((_, elem_type)) = value_info_to_type_vector(&output) else {
            continue;
        };
        if elem_type == 6 || elem_type == 7 {
            //INT32, INT64 は int* として表現
            sig_lines.push(format!("    int* {unique_name}_ptr,"));
        } else {
            // その他の型は float* として表現
            sig_lines.push(format!("    float* {unique_name}_ptr,"));
        }
    }

    lines.extend(sig_lines);

    Ok(lines.join("\n"))
}
