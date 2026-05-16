use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

pub struct ReduceSumOperator<'a> {
    // ReduceSum specific fields for the reduce sum operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for ReduceSumOperator<'a> {
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
        // Implementation for generating C++ code for the reduce sum operator
        let [input] = self.base_data().inputs.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "ReduceSum operator requires exactly 1 input"
            ));
        };
        let in_var = self.get_mapped_variable(input)?;
        let out_var = self.get_output_var_name();

        let axes = self.base_data().get_axes()?;
        let shape = self.in_shape(0)?;
        let [shape0, shape1] = shape.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "Input tensor must be 2D for ReduceSum operator"
            ));
        };
        self.set_output_var_name(out_var.clone())?;

        match axes.as_slice() {
            [0] => Ok(vec![format!(
                "    const Vector<{shape1}> {out_var} = col_sum<{shape0}, {shape1}>({in_var});"
            )]),
            [-1] | [1] => Ok(vec![format!(
                "    const Vector<{shape0}> {out_var} = row_sum<{shape0}, {shape1}>({in_var});"
            )]),
            _ => Err(color_eyre::eyre::eyre!(
                "Only reduction along axis 0 or 1 is supported for ReduceSum operator"
            )),
        }
    }
}
