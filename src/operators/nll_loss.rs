use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::base::{BaseData, BaseOperator};

pub struct NLLLossOperator<'a> {
    // NLLLoss specific fields for the NLLLoss operator if needed
    base_data: BaseData<'a>,
}

impl<'a> BaseOperator<'a> for NLLLossOperator<'a> {
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
                "NLLLoss operator requires exactly 2 inputs"
            ));
        };
        let log_prods_var = self.get_mapped_variable(input0);
        let target_var = self.get_mapped_variable(input1);
        let out_var = self.get_output_var_name();

        let log_prods_shape = self.in_shape(0)?;
        let target_shape = self.in_shape(1)?;
        let [batch_size, num_classes] = log_prods_shape.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "The first input tensor must be 2D for NLLLoss operator"
            ));
        };
        let [target_dim] = target_shape.as_slice() else {
            return Err(color_eyre::eyre::eyre!(
                "The second input tensor must be 1D for NLLLoss operator"
            ));
        };
        if target_dim != batch_size {
            return Err(color_eyre::eyre::eyre!(
                "The dimension of the second input tensor must match the batch size of the first input tensor for NLLLoss operator"
            ));
        }

        let reduction = self
            .base_data()
            .attr_iter()
            .find(|attr| attr.name == "reduction")
            .and_then(|attr| String::from_utf8(attr.s.clone()).ok())
            .unwrap_or("mean".to_string());
        self.set_output_var_name(out_var.clone())?;
        if reduction == "mean" {
            // gather_sum関数を使用
            Ok(vec![format!(
                "    const float {out_var} = gather_sum<{batch_size}, {num_classes}>(mul_constant<{batch_size}, {num_classes}>({log_prods_var}, -1.0f / {batch_size}), {target_var});"
            )])
        } else {
            Err(color_eyre::eyre::eyre!(
                "Only 'mean' reduction is supported for NLLLoss operator"
            ))
        }
    }
}
