use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::*;
use crate::utils::value_info_to_type_vector;

pub struct BaseData<'a> {
    pub node_proto: NodeProto,
    pub graph: GraphProto,
    pub var_map: &'a mut HashMap<String, String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub name: String,
}

impl<'a> BaseData<'a> {
    pub fn new(
        node_proto: &NodeProto,
        graph: &GraphProto,
        var_map: &'a mut HashMap<String, String>,
    ) -> Self {
        let inputs = node_proto.input.clone();
        let outputs = node_proto.output.clone();
        let name = node_proto.name.clone();
        Self {
            node_proto: node_proto.clone(),
            graph: graph.clone(),
            var_map,
            inputs,
            outputs,
            name,
        }
    }

    pub fn attr_iter(&self) -> impl Iterator<Item = &tract_onnx::pb::AttributeProto> {
        self.node_proto.attribute.iter()
    }

    pub fn get_axes(&self) -> Result<Vec<i64>> {
        let axes_attr = self
            .attr_iter()
            .find(|attr| attr.name == "axes")
            .ok_or_else(|| {
                color_eyre::eyre::eyre!("Attribute 'axes' not found for operator {}", self.name)
            })?;
        Ok(axes_attr.ints.to_vec())
    }

    pub fn initializer_iter(&self) -> impl Iterator<Item = &tract_onnx::pb::TensorProto> {
        self.graph.initializer.iter()
    }
}

pub trait BaseOperator<'a> {
    fn new(
        node_proto: &NodeProto,
        graph: &GraphProto,
        var_map: &'a mut HashMap<String, String>,
    ) -> Self
    where
        Self: Sized;

    fn base_data_mut(&mut self) -> &mut BaseData<'a>;
    fn base_data(&self) -> &BaseData<'a>;

    fn generate_cpp_code(&mut self) -> Result<Vec<String>>;

    fn get_mapped_variable(&self, original_name: &str) -> String {
        self.base_data()
            .var_map
            .get(original_name)
            .unwrap_or(&original_name.to_string())
            .clone()
    }

    fn get_output_var_name(&self) -> String {
        self.base_data()
            .attr_iter()
            .find(|attr| attr.name == "var_name")
            .and_then(|attr| {
                // attr.sはVec<u8>なので、Stringに変換する必要がある
                String::from_utf8(attr.s.clone()).ok()
            })
            .unwrap_or(self.base_data().name.clone())
    }

    fn in_shape(&self, index: usize) -> Result<Vec<usize>> {
        let input_name = self.base_data().inputs.get(index).ok_or_else(|| {
            color_eyre::eyre::eyre!(
                "Input index {} out of bounds for operator {}",
                index,
                self.base_data().name
            )
        })?;
        self.get_tensor_shape(input_name)
    }

    fn get_tensor_shape(&self, tensor_name: &str) -> Result<Vec<usize>> {
        // まず、グラフのinput、value_info、output、initializerを順番に検索するために一連の長いchainを作成する
        if let Some(shape) = [
            self.base_data().graph.input.iter(),
            self.base_data().graph.value_info.iter(),
            self.base_data().graph.output.iter(),
        ]
        .into_iter()
        .flatten()
        .find(|value_info| value_info.name == tensor_name)
        .and_then(|value_info| value_info_to_type_vector(value_info).map(|(shape, _)| shape))
        {
            Ok(shape)
        } else if let Some(shape) = self
            .base_data()
            .initializer_iter()
            .find(|init| init.name == tensor_name)
            .map(|init| init.dims.iter().map(|d| *d as usize).collect())
        {
            Ok(shape)
        } else {
            Err(color_eyre::eyre::eyre!(
                "Tensor shape not found for tensor: {}",
                tensor_name
            ))
        }
    }

    fn set_output_var_name(&mut self, var_name: String) -> Result<()> {
        // var_mapに出力変数名を追加
        let output_name = self
            .base_data()
            .outputs
            .first()
            .ok_or_else(|| {
                color_eyre::eyre::eyre!("No outputs found for operator {}", self.base_data().name)
            })?
            .clone();
        self.base_data_mut()
            .var_map
            .insert(output_name.clone(), var_name);
        Ok(())
    }
}

pub fn gen_base_operator<'a>(
    node_proto: &NodeProto,
    graph: &GraphProto,
    var_map: &'a mut HashMap<String, String>,
) -> Result<Box<dyn BaseOperator<'a> + 'a>> {
    match node_proto.op_type.as_str() {
        "Add" => Ok(Box::new(AddOperator::new(node_proto, graph, var_map))),
        "Gemm" => Ok(Box::new(GemmOperator::new(node_proto, graph, var_map))),
        "Relu" => Ok(Box::new(ReluOperator::new(node_proto, graph, var_map))),
        "Identity" => Ok(Box::new(IdentityOperator::new(node_proto, graph, var_map))),
        "ReduceMax" => Ok(Box::new(ReduceMaxOperator::new(node_proto, graph, var_map))),
        "Sub" => Ok(Box::new(SubOperator::new(node_proto, graph, var_map))),
        "Exp" => Ok(Box::new(ExpOperator::new(node_proto, graph, var_map))),
        "ReduceSum" => Ok(Box::new(ReduceSumOperator::new(node_proto, graph, var_map))),
        "Div" => Ok(Box::new(DivOperator::new(node_proto, graph, var_map))),
        "Log" => Ok(Box::new(LogOperator::new(node_proto, graph, var_map))),
        "NegativeLogLikelihoodLoss" => {
            Ok(Box::new(NLLLossOperator::new(node_proto, graph, var_map)))
        }
        "OneHot" => Ok(Box::new(OneHotOperator::new(node_proto, graph, var_map))),
        "Mul" => Ok(Box::new(MulOperator::new(node_proto, graph, var_map))),
        "Greater" => Ok(Box::new(GreaterOperator::new(node_proto, graph, var_map))),
        "Cast" => Ok(Box::new(CastOperator::new(node_proto, graph, var_map))),
        _ => Err(color_eyre::eyre::eyre!(
            "Unsupported operator type: {}",
            node_proto.op_type
        )),
    }
}
