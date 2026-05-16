mod base;

mod add;
mod gemm;
mod relu;

use add::AddOperator;
use gemm::GemmOperator;
use relu::ReluOperator;

pub use base::gen_base_operator;
