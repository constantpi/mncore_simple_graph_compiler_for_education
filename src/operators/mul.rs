use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};
use crate::utils::tensor_proto_to_float_vector;

pub struct MulOperator<'a> {
    // Mul specific fields for the mul operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for MulOperator<'a> {
    fn new(
        node_proto: &NodeProto,
        graph: &GraphProto,
        var_map: &'a mut HashMap<String, String>,
    ) -> Self {
        Self {
            base_data: BaseData::new(node_proto, graph, var_map),
        }
    }

    fn base_data(&self) -> &BaseData<'a> {
        &self.base_data
    }
    fn base_data_mut(&mut self) -> &mut BaseData<'a> {
        &mut self.base_data
    }

    fn generate_cpp_code(&mut self) -> Result<Vec<String>> {
        let [input0, input1] = self.base_data().inputs.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "Mul operator requires exactly 2 inputs"
            ));
        };
        println!("input0: {input0}, input1: {input1}");
        let in_var0 = self.get_mapped_variable(input0);
        let in_var1 = self.get_mapped_variable(input1);
        let out_var = self.get_output_var_name();

        let shape0 = self.in_shape(0)?;
        let shape1 = self.in_shape(1)?;

        // 2番めの入力が初期化子かどうかをチェック
        let is_initializer = self
            .base_data()
            .initializer_iter()
            .any(|init| &init.name == input1);

        // 2番めの入力がスカラーかどうかをチェック
        let is_scalar = shape1.is_empty();
        let code = if is_scalar {
            let scalar_value = if is_initializer {
                // 初期化子でスカラーの場合、定数としてコードを生成
                let tensor = self
                    .base_data()
                    .initializer_iter()
                    .find(|init| &init.name == input1)
                    .and_then(tensor_proto_to_float_vector)
                    .ok_or(color_eyre::eyre::eyre!(
                        "Failed to extract scalar value from initializer for input {input1}"
                    ))?;
                let [coeff] = tensor.as_slice() else {
                    return Err(color_eyre::eyre::eyre!(
                        "Expected a single scalar value in the initializer for input {input1}"
                    ));
                };
                format!("{coeff}f")
            } else {
                in_var1
            };
            match shape0.as_slice() {
                [dim0, dim1] => format!(
                    "    const Matrix<{dim0}, {dim1}> {out_var} = mul_constant<{dim0}, {dim1}>({in_var0}, {scalar_value});"
                ),
                [dim0] => format!(
                    "    const Vector<{dim0}> {out_var} = mul_constant<{dim0}>({in_var0}, {scalar_value});"
                ),
                _ => {
                    return Err(color_eyre::eyre::eyre!(
                        "Unsupported input shape for Mul operator when the second input is a scalar: {:?}",
                        shape0
                    ));
                }
            }
        } else {
            // 要素ごとの乗算
            match (shape0.as_slice(), shape1.as_slice()) {
                ([dim0, dim1], [dim0_2, dim1_2]) if dim0 == dim0_2 && dim1 == dim1_2 => format!(
                    "    const Matrix<{dim0}, {dim1}> {out_var} = mul_elem<{dim0}, {dim1}>({in_var0}, {in_var1});"
                ),
                _ => {
                    return Err(color_eyre::eyre::eyre!(
                        "Unsupported input shapes for Mul operator: {:?} and {:?}",
                        shape0,
                        shape1
                    ));
                }
            }
        };

        self.set_output_var_name(out_var)?;
        Ok(vec![code])
    }
}
