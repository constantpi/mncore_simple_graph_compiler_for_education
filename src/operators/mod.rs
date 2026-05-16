mod base;

mod add;
mod div;
mod exp;
mod gemm;
mod identity;
mod log;
mod mul;
mod nll_loss;
mod onehot;
mod reduce_max;
mod reduce_sum;
mod relu;
mod sub;

use add::AddOperator;
use div::DivOperator;
use exp::ExpOperator;
use gemm::GemmOperator;
use identity::IdentityOperator;
use log::LogOperator;
use mul::MulOperator;
use nll_loss::NLLLossOperator;
use onehot::OneHotOperator;
use reduce_max::ReduceMaxOperator;
use reduce_sum::ReduceSumOperator;
use relu::ReluOperator;
use sub::SubOperator;

pub use base::gen_base_operator;
