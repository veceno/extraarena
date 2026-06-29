//! Training-only Rust scaffolding for TrainV3.
//!
//! Python `core/engine.py` remains the production and parity oracle. This crate is
//! intentionally isolated under `TrainV3/` and is not imported by production bot code.

pub mod action_codec;
pub mod card_shape;
pub mod env;
pub mod exploit;
pub mod ffi;
pub mod kernel;
pub mod ppo;
pub mod reward;
pub mod state;
pub mod trace;
pub mod v5;
pub mod worker;

pub const OBS_DIM_V1: usize = 1456;
pub const MAX_CANDIDATE_ACTIONS: usize = 601;
pub const ACTION_FEATURE_DIM_V1: usize = 171;
pub const CARD_SHAPE_DIM: usize = 64;
