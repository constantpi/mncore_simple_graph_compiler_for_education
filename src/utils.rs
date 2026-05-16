use tract_onnx::pb::{ValueInfoProto, tensor_shape_proto::dimension, type_proto};

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
