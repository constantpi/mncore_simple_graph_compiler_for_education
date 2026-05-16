use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

pub struct DivOperator<'a> {
    // Div specific fields for the div operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for DivOperator<'a> {
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
                "Div operator requires exactly 2 inputs"
            ));
        };
        let in_var1 = self.get_mapped_variable(input1)?;
        let in_var2 = self.get_mapped_variable(input2)?;
        let out_var = self.get_output_var_name();

        let shape1 = self.in_shape(0)?;
        let shape2 = self.in_shape(1)?;
        let ([dim0, dim1], [dim2]) = (shape1.as_slice(), shape2.as_slice()) else {
            return Err(color_eyre::eyre::eyre!(
                "Input tensors must be 2D and 1D for Div operator"
            ));
        };
        if dim0 != dim2 {
            return Err(color_eyre::eyre::eyre!(
                "The first dimension of the first input must match the dimension of the second input for Div operator"
            ));
        }

        self.set_output_var_name(out_var.clone())?;
        Ok(vec![format!(
            "    const Matrix<{dim0}, {dim1}> {out_var} = div_rowvec<{dim0}, {dim1}>({in_var1}, {in_var2});"
        )])
    }
}
