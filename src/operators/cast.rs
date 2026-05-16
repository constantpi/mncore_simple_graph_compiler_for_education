use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

pub struct CastOperator<'a> {
    // Cast specific fields for the Cast operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for CastOperator<'a> {
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
                "Cast operator requires exactly 1 input"
            ));
        };
        let in_var = self.get_mapped_variable(input);
        let _out_var = self.get_output_var_name();

        let to_type = self
            .base_data()
            .attr_iter()
            .find(|attr| attr.name == "to")
            .map(|attr| attr.i);
        self.set_output_var_name(in_var)?;

        if to_type == Some(1) {
            // Cast to float
            Ok(vec![])
        } else {
            Err(color_eyre::eyre::eyre!(
                "Unsupported target type for Cast operator: {:?}",
                to_type
            ))
        }
    }
}
