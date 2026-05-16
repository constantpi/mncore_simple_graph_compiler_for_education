use color_eyre::eyre::Result;
use std::{collections::HashMap, fmt::format};
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

// Gemm演算オペレーター: Y = alpha * A @ B^T（バイアスなし、転置のみサポート）
pub struct GemmOperator<'a> {
    // Gemm specific fields for the gemm operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for GemmOperator<'a> {
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
                "Gemm operator requires exactly 2 inputs"
            ));
        };

        let in_a = self.get_mapped_variable(input0)?;
        let in_b = self.get_mapped_variable(input1)?;
        let out_var = self.get_output_var_name();

        if self
            .base_data
            .attr_iter()
            .find(|attr| attr.name == "alpha" && attr.f != 1.0)
            .is_some()
        {
            return Err(color_eyre::eyre::eyre!(
                "Gemm operator with alpha != 1.0 is not supported"
            ));
        }
        if self
            .base_data
            .attr_iter()
            .find(|attr| attr.name == "beta" && attr.f != 0.0)
            .is_some()
        {
            return Err(color_eyre::eyre::eyre!(
                "Gemm operator with beta != 0.0 is not supported"
            ));
        }
        let trans_a = self
            .base_data
            .attr_iter()
            .find(|attr| attr.name == "transA")
            .map(|attr| attr.i != 0)
            .unwrap_or(false);
        let trans_b = self
            .base_data
            .attr_iter()
            .find(|attr| attr.name == "transB")
            .map(|attr| attr.i != 0)
            .unwrap_or(false);

        let [shape_a_0, shape_a_1] = self
            .in_shape(0)?
            .try_into()
            .map_err(|_| color_eyre::eyre::eyre!("Invalid shape for input 0"))?;
        let [shape_b_0, shape_b_1] = self
            .in_shape(1)?
            .try_into()
            .map_err(|_| color_eyre::eyre::eyre!("Invalid shape for input 1"))?;

        let (m, k_a, in_a) = if trans_a {
            (
                shape_a_1,
                shape_a_0,
                format!("trans<{shape_a_0},{shape_a_1}>({in_a})"),
            )
        } else {
            (shape_a_0, shape_a_1, in_a)
        };
        let (k_b, n, in_b) = if trans_b {
            (
                shape_b_1,
                shape_b_0,
                format!("trans<{shape_b_0},{shape_b_1}>({in_b})"),
            )
        } else {
            (shape_b_0, shape_b_1, in_b)
        };

        if k_a != k_b {
            return Err(color_eyre::eyre::eyre!(
                "Inner dimensions of A and B must match for Gemm operator"
            ));
        }

        self.set_output_var_name(out_var.clone())?;
        Ok(vec![format!(
            "    const Matrix<{m}, {n}> {out_var} = matmul<{m}, {k_a}, {n}>({in_a}, {in_b});"
        )])
    }
}
