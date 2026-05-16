use color_eyre::eyre::Result;
use std::collections::HashMap;
use tract_onnx::pb::{GraphProto, NodeProto};

use super::AddOperator;

pub struct BaseData {
    pub node_proto: NodeProto,
    pub graph: GraphProto,
    pub var_map: HashMap<String, String>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub name: String,
}

impl BaseData {
    pub fn new(
        node_proto: &NodeProto,
        graph: &GraphProto,
        var_map: &mut HashMap<String, String>,
    ) -> Self {
        let inputs = node_proto.input.clone();
        let outputs = node_proto.output.clone();
        let name = node_proto.name.clone();
        Self {
            node_proto: node_proto.clone(),
            graph: graph.clone(),
            var_map: var_map.clone(),
            inputs,
            outputs,
            name,
        }
    }
}

pub trait BaseOperator {
    fn new(
        node_proto: &NodeProto,
        graph: &GraphProto,
        var_map: &mut HashMap<String, String>,
    ) -> Self
    where
        Self: Sized;

    fn base_data_mut(&mut self) -> &mut BaseData;
    fn base_data(&self) -> &BaseData;

    fn generate_cpp_code(&mut self) -> Result<String>;

    fn get_mapped_variable(&self, original_name: &str) -> Result<String> {
        self.base_data()
            .var_map
            .get(original_name)
            .cloned()
            .ok_or_else(|| {
                color_eyre::eyre::eyre!("Variable {} not found in var_map", original_name)
            })
    }

    fn get_output_var_name(&self) -> String {
        self.base_data()
            .node_proto
            .attribute
            .iter()
            .find(|attr| attr.name == "var_name")
            .and_then(|attr| {
                // attr.sはVec<u8>なので、Stringに変換する必要がある
                String::from_utf8(attr.s.clone()).ok()
            })
            .unwrap_or(self.base_data().name.clone())
    }
}

pub fn gen_base_operator(
    node_proto: &NodeProto,
    graph: &GraphProto,
    var_map: &mut HashMap<String, String>,
) -> Result<Box<dyn BaseOperator>> {
    match node_proto.op_type.as_str() {
        // "Add" => Box::new(AddOperator::new(node_proto, graph, var_map)),
        _ => Ok(Box::new(AddOperator::new(node_proto, graph, var_map))),
        // 他の演算子もここに追加
        _ => Err(color_eyre::eyre::eyre!(
            "Unsupported operator type: {}",
            node_proto.op_type
        )),
    }
}
