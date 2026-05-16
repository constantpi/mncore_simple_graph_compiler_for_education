use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

pub struct ReduceMaxOperator<'a> {
    // ReduceMax specific fields for the reduce_max operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for ReduceMaxOperator<'a> {
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
        let [input] = self.base_data().inputs.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "ReduceMax operator requires exactly 1 input"
            ));
        };
        let in_var = self.get_mapped_variable(input)?;
        let out_var = self.get_output_var_name();

        let axes = self.base_data().get_axes()?;
        let shape = self.in_shape(0)?;
        let [shape0, shape1] = shape.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "Input tensor must be 2D for ReduceMax operator"
            ));
        };
        if axes != [-1] {
            return Err(color_eyre::eyre::eyre!(
                "Only reduction along axis 0 is supported for ReduceMax operator"
            ));
        }

        self.set_output_var_name(out_var.clone())?;
        Ok(vec![format!(
            "    const Vector<{shape0}> {out_var} = row_max<{shape0}, {shape1}>({in_var});"
        )])
    }
}
