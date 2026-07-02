pub const GOLDEN_TRACE_SCHEMA: &str = "trainv3-golden-trace-v1";

#[derive(Debug, Clone)]
pub struct TraceHashSet {
    pub state_sha256: String,
    pub observation_sha256_f32_le: String,
    pub mask_sha256_f32_le: String,
    pub action_features_sha256_f32_le: Option<String>,
}
