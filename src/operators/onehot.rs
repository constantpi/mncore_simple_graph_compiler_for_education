use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};
use crate::utils::tensor_proto_to_int_vector;

pub struct OneHotOperator<'a> {
    // OneHot specific fields for the OneHot operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for OneHotOperator<'a> {
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
        let [input, ..] = self.base_data().inputs.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "OneHot operator requires exactly 1 input"
            ));
        };
        let in_var = self.get_mapped_variable(input);
        let out_var = self.get_output_var_name();

        let in_shape = self.in_shape(0)?;
        let [batch_size] = in_shape.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "Input tensor must be 1D for OneHot operator"
            ));
        };
        let depth = self.get_depth()?;
        self.set_output_var_name(out_var.clone())?;

        // OneHot変換（ループで実装）
        let lines = vec![
            format!(
                "    Matrix<{batch_size}, {depth}> {out_var} = zeros<{batch_size}, {depth}>();"
            ),
            format!("    for (int i = 0; i < {batch_size}; i++) {{"),
            format!("        {out_var}[i][{in_var}[i]] = 1.0f;"),
            format!("    }}"),
        ];

        // lines.append(f"    Matrix<{batch_size}, {depth}> {out_var} = zeros<{batch_size}, {depth}>();")
        // lines.append(f"    for (int i = 0; i < {batch_size}; i++) {{")
        // lines.append(f"        {out_var}[i][{indices_var}[i]] = 1.0f;")
        // lines.append(f"    }}")
        Ok(lines)
    }
}

impl OneHotOperator<'_> {
    fn get_depth(&self) -> Result<usize> {
        if let Some(depth) = self
            .base_data()
            .attr_iter()
            .find(|attr| attr.name == "depth")
            .map(|attr| attr.i)
        {
            Ok(depth as usize)
        } else {
            let [_, depth_name] = self.base_data().inputs.as_slice() else {
                return Err(color_eyre::eyre::eyre!(
                    "OneHot operator requires a second input for depth if the 'depth' attribute is not provided"
                ));
            };
            let Some(tensor) = self
                .base_data()
                .initializer_iter()
                .find(|init| &init.name == depth_name)
                .and_then(tensor_proto_to_int_vector)
            else {
                return Err(color_eyre::eyre::eyre!(
                    "The second input for depth must be an initializer tensor for OneHot operator"
                ));
            };
            let [depth] = tensor.as_slice() else {
                return Err(color_eyre::eyre::eyre!(
                    "The depth tensor must be a scalar for OneHot operator"
                ));
            };
            Ok(*depth as usize)
        }
    }
}
