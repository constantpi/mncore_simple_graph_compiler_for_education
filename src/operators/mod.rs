mod base;

mod add;
mod gemm;
mod identity;
mod reduce_max;
mod relu;

use add::AddOperator;
use gemm::GemmOperator;
use identity::IdentityOperator;
use reduce_max::ReduceMaxOperator;
use relu::ReluOperator;

pub use base::gen_base_operator;
