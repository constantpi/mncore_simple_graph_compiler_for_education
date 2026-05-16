use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

pub struct IdentityOperator<'a> {
    // Identity specific fields for the identity operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for IdentityOperator<'a> {
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
                "Identity operator requires exactly 1 input"
            ));
        };
        self.set_output_var_name(input.to_string())?;
        Ok(vec![])
    }
}
