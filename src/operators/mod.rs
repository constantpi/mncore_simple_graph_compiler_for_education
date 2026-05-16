mod base;

mod add;
mod gemm;
mod identity;
mod reduce_max;
mod relu;
mod sub;

use add::AddOperator;
use gemm::GemmOperator;
use identity::IdentityOperator;
use reduce_max::ReduceMaxOperator;
use relu::ReluOperator;
use sub::SubOperator;

pub use base::gen_base_operator;
