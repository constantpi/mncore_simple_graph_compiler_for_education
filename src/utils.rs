use tract_onnx::pb::{
    TensorProto, ValueInfoProto, tensor_proto::DataType, tensor_shape_proto::dimension, type_proto,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    Float, // float
    Int,   // int
}

pub fn value_info_to_type_vector(value_info: &ValueInfoProto) -> Option<(Vec<usize>, ElemType)> {
    let Some(type_proto) = &value_info.r#type else {
        return None;
    };
    let Some(tensor_type) = &type_proto.value else {
        return None;
    };
    let type_proto::Value::TensorType(tensor) = tensor_type;
    let elem_type = tensor.elem_type;
    let Some(shape) = &tensor.shape else {
        return None;
    };
    let value_dim = shape
        .dim
        .iter()
        .map(|d| d.value.clone())
        .collect::<Option<Vec<_>>>()?;
    let dim = value_dim
        .iter()
        .map(|d| {
            if let dimension::Value::DimValue(size) = d {
                Some(*size as usize)
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;

    let elem_type = match elem_type {
        6 => ElemType::Int,   // INT32
        7 => ElemType::Int,   // INT64
        _ => ElemType::Float, // それ以外はすべてfloatとして扱う
    };
    Some((dim, elem_type))
}

pub fn tensor_proto_to_dim_vector(tensor_proto: &TensorProto) -> Vec<usize> {
    tensor_proto.dims.iter().map(|d| *d as usize).collect()
}

fn calc_elem_count(dims: &[usize]) -> usize {
    dims.iter().product()
}

pub fn tensor_proto_to_int_vector(tensor_proto: &TensorProto) -> Option<Vec<i64>> {
    let data_type = tensor_proto.data_type();
    let ans = match data_type {
        DataType::Int32 => tensor_proto
            .int32_data
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>(),
        DataType::Int64 => tensor_proto.int64_data.iter().cloned().collect(),
        _ => return None,
    };
    if ans.len() == calc_elem_count(&tensor_proto_to_dim_vector(tensor_proto)) {
        Some(ans)
    } else {
        None
    }
}

pub fn tensor_proto_to_float_vector(tensor_proto: &TensorProto) -> Option<Vec<f64>> {
    let data_type = tensor_proto.data_type();
    let ans = match data_type {
        DataType::Float => tensor_proto
            .float_data
            .iter()
            .map(|x| *x as f64)
            .collect::<Vec<_>>(),
        DataType::Double => tensor_proto.double_data.iter().cloned().collect(),
        _ => return None,
    };
    if ans.len() == calc_elem_count(&tensor_proto_to_dim_vector(tensor_proto)) {
        Some(ans)
    } else {
        None
    }
}
