use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};
use crate::utils::tensor_proto_to_float_vector;

pub struct GreaterOperator<'a> {
    // Greater specific fields for the Greater operator if needed
    base_data: BaseData<'a>,
}
impl<'a> BaseOperator<'a> for GreaterOperator<'a> {
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
        let [input1, input2] = self.base_data().inputs.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "Greater operator requires exactly 2 inputs"
            ));
        };
        let in_var1 = self.get_mapped_variable(input1);
        let out_var = self.get_output_var_name();

        let in_shape1 = self.in_shape(0)?;
        let in_shape2 = self.in_shape(1)?;
        if !in_shape2.is_empty() {
            return Err(color_eyre::eyre::eyre!(
                "Second input tensor must be empty for Greater operator"
            ));
        }

        let is_zero = self
            .base_data()
            .initializer_iter()
            .find(|init| &init.name == input2)
            .and_then(tensor_proto_to_float_vector)
            .and_then(|vec| Some(vec == vec![0.0]))
            .unwrap_or(false);
        if !is_zero {
            return Err(color_eyre::eyre::eyre!(
                "Second input tensor must be a constant zero for Greater operator"
            ));
        }

        let [dim1, dim2] = in_shape1.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "Input tensor must be 2D for Greater operator"
            ));
        };

        self.set_output_var_name(out_var.clone())?;

        // relu_gradを使用: relu_grad(grad, x)はx > 0の場所でgradを返し、それ以外で0を返す
        // 1で埋めた行列を作成してgradとして使用
        let lines = vec![
            format!("    Matrix<{dim1}, {dim2}> ones = zeros<{dim1}, {dim2}>();"),
            format!("    for (int i = 0; i < {dim1}; i++) {{"),
            format!("        for (int j = 0; j < {dim2}; j++) {{"),
            format!("            ones[i][j] = 1.0f;"),
            format!("        }}"),
            format!("    }}"),
            format!(
                "    const Matrix<{dim1}, {dim2}> {out_var} = relu_grad<{dim1}, {dim2}>(ones, {in_var1});"
            ),
        ];
        Ok(lines)
    }
}
