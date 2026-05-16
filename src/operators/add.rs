use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

pub struct AddOperator {
    // Add specific fields for the add operator if needed
    base_data: BaseData,
}

impl BaseOperator for AddOperator {
    fn new(
        node_proto: &NodeProto,
        graph: &GraphProto,
        var_map: &mut HashMap<String, String>,
    ) -> Self {
        Self {
            base_data: BaseData::new(node_proto, graph, var_map),
        }
    }

    fn base_data(&self) -> &BaseData {
        &self.base_data
    }
    fn base_data_mut(&mut self) -> &mut BaseData {
        &mut self.base_data
    }

    fn generate_cpp_code(&mut self) -> Result<Vec<String>> {
        // Implementation for generating C++ code for the add operator
        let [input0, input1] = self.base_data().inputs.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "Add operator requires exactly 2 inputs"
            ));
        };

        let in0 = self.get_mapped_variable(input0)?;
        let in1 = self.get_mapped_variable(input1)?;
        let out_var = self.get_output_var_name();
        self.set_output_var_name(out_var.clone())?;

        let shape0 = self.in_shape(0)?;
        let shape1 = self.in_shape(1)?;

        let ([dim0, dim1], [dim2]) = (shape0.as_slice(), shape1.as_slice()) else {
            return Err(color_eyre::eyre::eyre!("Add operator requires 2D inputs"));
        };
        if dim1 != dim2 {
            return Err(color_eyre::eyre::eyre!(
                "Add operator requires inputs with the same shape"
            ));
        }
        Ok(vec![
            format!(
                "    const Matrix<{r}, {c}> {out_var} = add_colvec<{r}, {c}>({in0}, {in1});",
                r = dim0,
                c = dim1
            )
            .to_string(),
        ])
    }
}
