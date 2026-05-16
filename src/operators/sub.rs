use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

pub struct SubOperator<'a> {
    // Sub specific fields for the sub operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for SubOperator<'a> {
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
        // Implementation for generating C++ code for the sub operator
        let [input0, input1] = self.base_data().inputs.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "Sub operator requires exactly 2 inputs"
            ));
        };

        let in0 = self.get_mapped_variable(input0);
        let in1 = self.get_mapped_variable(input1);
        let out_var = self.get_output_var_name();

        let shape0 = self.in_shape(0)?;
        let shape1 = self.in_shape(1)?;

        self.set_output_var_name(out_var.clone())?;

        match (shape0.as_slice(), shape1.as_slice()) {
            ([dim0, dim1], [dim2]) if dim0 == dim2 => Ok(vec![
                format!(
                    "    const Matrix<{r}, {c}> {out_var} = sub_rowvec<{r}, {c}>({in0}, {in1});",
                    r = dim0,
                    c = dim1
                )
                .to_string(),
            ]),
            ([dim0], [dim1]) if dim0 == dim1 => Ok(vec![
                format!(
                    "    const Vector<{d}> {out_var} = sub<{d}>({in0}, {in1});",
                    d = dim0
                )
                .to_string(),
            ]),
            ([dim0, dim1], [dim2, dim3]) if dim0 == dim2 && dim1 == dim3 => Ok(vec![
                format!(
                    "    const Matrix<{r}, {c}> {out_var} = sub<{r}, {c}>({in0}, {in1});",
                    r = dim0,
                    c = dim1
                )
                .to_string(),
            ]),
            _ => Err(color_eyre::eyre::eyre!(
                "Sub operator requires 2D inputs with the same shape"
            )),
        }
    }
}
