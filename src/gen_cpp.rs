use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::ModelProto;

use crate::operators::gen_base_operator;
use crate::utils::{ElemType, value_info_to_type_vector};

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
        let Some((dim, _elem_type)) = value_info_to_type_vector(out) else {
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
        let Some((_dim, elem_type)) = value_info_to_type_vector(input) else {
            continue;
        };
        if elem_type == ElemType::Int {
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
        let Some((_, elem_type)) = value_info_to_type_vector(output) else {
            continue;
        };
        if elem_type == ElemType::Int {
            //INT32, INT64 は int* として表現
            sig_lines.push(format!("    int* {unique_name}_ptr,"));
        } else {
            // その他の型は float* として表現
            sig_lines.push(format!("    float* {unique_name}_ptr,"));
        }
    }
    // sig_linesの最後の行の末尾のカンマを削除して、関数シグネチャを完成させる
    if let Some(last_line) = sig_lines.last_mut()
        && last_line.ends_with(",")
    {
        last_line.pop();
    }
    sig_lines.push(") {".to_string());
    lines.extend(sig_lines);

    // 変数名の追跡
    let mut variable_map = HashMap::new();

    // すべての入力とパラメータを読み込む
    lines.push("    // 入力とパラメータを読み込む".to_string());
    for input in graph.input.iter() {
        let Some((dim, elem_type)) = value_info_to_type_vector(input) else {
            continue;
        };
        variable_map.insert(input.name.clone(), input.name.clone());
        match elem_type {
            ElemType::Int => {
                if dim.len() == 1 {
                    lines.push(format!(
                        "    const array {} = load<{}, int>({}_ptr);",
                        input.name, dim[0], input.name
                    ));
                } else {
                    return Err(color_eyre::eyre::eyre!(
                        "Unsupported input shape for int type: {:?}",
                        dim
                    ));
                }
            }
            ElemType::Float => {
                // # 通常のfloatテンソル
                match dim.len() {
                    2 => lines.push(format!(
                        "    const Matrix<{dim0}, {dim1}> {name} = load<{dim0}, {dim1}>({name}_ptr);",
                        dim0 = dim[0],
                        dim1 = dim[1],
                        name = input.name
                    )),
                    1 => lines.push(format!(
                        "    const Vector<{dim0}> {name} = load<{dim0}>({name}_ptr);",
                        dim0 = dim[0],
                        name = input.name
                    )),
                    0 => lines.push(format!(
                        "    const float {name} = *{name}_ptr;",
                        name = input.name
                    )),
                    _ => {
                        return Err(color_eyre::eyre::eyre!(
                            "Unsupported input shape for float type: {:?}",
                            dim
                        ));
                    }
                }
            }
        }
    }
    lines.push("".to_string());

    // ノードを処理
    lines.push("    // ノードを処理".to_string());
    for node in graph.node.iter() {
        let op_type = node.op_type.clone();
        println!("Processing node: {} of type {}", node.name, op_type);
        let mut operator = gen_base_operator(node, &graph, &mut variable_map)?;
        let op_lines = operator.generate_cpp_code()?;
        println!(
            "Generated code for node {}:\n{}",
            node.name,
            op_lines.join("\n")
        );
        lines.extend(op_lines);
    }

    Ok(lines.join("\n"))
}
