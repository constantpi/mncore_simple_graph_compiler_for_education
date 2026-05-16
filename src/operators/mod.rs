mod base;

mod add;
mod gemm;

use add::AddOperator;
use gemm::GemmOperator;

pub use base::gen_base_operator;
