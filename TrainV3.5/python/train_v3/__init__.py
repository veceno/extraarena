"""Training-only helpers for TrainV3 environment preparation."""

from .aux_models import (
    AssemblerCandidate,
    AssemblerDatasetRow,
    DeckMatchupEvaluator,
    DesirererDatasetRow,
    DrawAssistController,
    DrawDesirerer,
    DrawScore,
    build_assembler_rows_from_matchup_summaries,
    build_assembler_rows_from_v5_trace,
    build_desirerer_rows_from_v5_trace,
    evaluate_assembler_baseline,
    evaluate_desirerer_baseline,
    load_assembler_dataset,
    load_desirerer_dataset,
    save_assembler_dataset,
    save_assembler_dataset_with_manifest,
    save_desirerer_dataset,
    save_desirerer_dataset_with_manifest,
)
from .contracts import AssistModeV5, InfoModeV5, OBS_V5_DIM
from .env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig
from .golden_trace import build_golden_trace
from .league_v5 import (
    V5EpisodeModes,
    V5LeagueConfig,
    compare_adaptive_strength_monotonicity,
    evaluate_adaptive_strength_proxy,
    parse_v5_opponent_mix,
    sample_v5_episode_modes,
)
from .obs_v5 import encode_observation_v5
from .reward_v5 import (
    V5RewardWeights,
    compute_history_outcome_deltas_v5,
    compute_reward_components_v5,
    compute_weighted_reward_v5,
    reward_snapshot_v5,
)
from .trace_factory_v5 import (
    V5TraceScenario,
    generate_v5_trace_pool,
    group_v5_trace_pool_by_mode,
    load_v5_trace_pool_manifest,
    resolve_v5_trace_paths,
    select_v5_trace_paths_for_mode,
)
from .v5_artifacts import (
    AUX_DATASET_SCHEMA,
    LEAGUE_RUN_SCHEMA,
    TRACE_POOL_SCHEMA,
    AuxDatasetManifest,
    LeagueRunManifest,
    TracePoolEntry,
    TracePoolManifest,
    manifest_to_dict,
    read_manifest_json,
    write_manifest_json,
)
from .rust_benchmark import (
    benchmark_compact_legal_policy_inference,
    benchmark_rust_gae_prepare,
    benchmark_rust_pre_step_action_tape_batch_modes,
    benchmark_rust_ppo_update_modes,
    benchmark_trainv3_speed_report,
    benchmark_rust_vec_collector_modes,
    benchmark_rust_vec_policy_collector_modes,
)
from .rust_collector import (
    RustTransitionBatch,
    collect_rust_vec_rollout,
    transition_batch_from_action_tape_rollout,
)
from .rust_ffi import (
    RustBatchWorker,
    RustCompactArgmaxActions,
    RustDenseArgmaxActions,
    RustPaddedArgmaxActions,
    RustPaddedLegalActions,
    RustPackedLegalRows,
    RustPreparedPPOBatch,
    compute_rust_compact_argmax_actions,
    compute_rust_dense_argmax_actions,
    compute_rust_gae_returns,
    compute_rust_pad_legal_actions,
    compute_rust_pack_legal_action_rows,
    compute_rust_padded_argmax_actions,
    compute_rust_prepare_ppo_batch,
    compute_rust_repeat_row_indices,
    compute_rust_selected_local_indices,
)
from .rust_policy import (
    CompactLegalActionScores,
    PaddedLegalActionScores,
    compact_argmax_actions,
    make_compact_legal_argmax_policy,
    make_dense_argmax_policy,
    make_padded_legal_argmax_policy,
    padded_argmax_actions,
    score_compact_legal_actions,
    score_padded_legal_actions,
)
from .rust_ppo import (
    RustPPOBatch,
    RustPPOEvaluation,
    evaluate_dense_rust_ppo_batch,
    evaluate_rust_ppo_batch,
    prepare_rust_ppo_batch,
    train_dense_rust_ppo_minibatch,
    train_rust_ppo_minibatch,
)
from .rust_rollout import RustRolloutStats, RustTraceRolloutRunner, benchmark_trace_file
from .rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_file, train_rust_ppo_trace_files
from .rust_vec_env import RustVecEnv, RustVecEnvReset, RustVecEnvStep


def __getattr__(name):
    if name == "run_v5_adaptive_training_pipeline":
        from .train_v5_adaptive import run_v5_adaptive_training_pipeline

        return run_v5_adaptive_training_pipeline
    if name == "train_v5_adaptive_main":
        from .train_v5_adaptive import main

        return main
    if name in {"V5ActionConditionedPolicy", "create_v5_policy"}:
        from .v5_policy import V5ActionConditionedPolicy, create_v5_policy

        return {"V5ActionConditionedPolicy": V5ActionConditionedPolicy, "create_v5_policy": create_v5_policy}[name]
    if name in {"OpenAICompatibleTeacherClient", "OpenAICompatibleTeacherConfig", "TeacherPreferenceRow"}:
        from .llm_teacher import (
            OpenAICompatibleTeacherClient,
            OpenAICompatibleTeacherConfig,
            TeacherPreferenceRow,
        )

        return {
            "OpenAICompatibleTeacherClient": OpenAICompatibleTeacherClient,
            "OpenAICompatibleTeacherConfig": OpenAICompatibleTeacherConfig,
            "TeacherPreferenceRow": TeacherPreferenceRow,
        }[name]
    if name in {"V5GauntletConfig", "ExploitLaneConfig", "build_default_exploit_gauntlet"}:
        from .gauntlet_v5 import ExploitLaneConfig, V5GauntletConfig, build_default_exploit_gauntlet

        return {
            "V5GauntletConfig": V5GauntletConfig,
            "ExploitLaneConfig": ExploitLaneConfig,
            "build_default_exploit_gauntlet": build_default_exploit_gauntlet,
        }[name]
    if name in {
        "V5OpponentLane",
        "assert_phase9_broad_environment_ready",
        "build_phase9_broad_opponent_lanes",
        "phase9_broad_opponent_mix",
        "prepare_phase9_broad_opponent_environment",
        "validate_broad_opponent_lanes",
    }:
        from .opponents_v5 import (
            V5OpponentLane,
            assert_phase9_broad_environment_ready,
            build_phase9_broad_opponent_lanes,
            phase9_broad_opponent_mix,
            prepare_phase9_broad_opponent_environment,
            validate_broad_opponent_lanes,
        )

        return {
            "V5OpponentLane": V5OpponentLane,
            "assert_phase9_broad_environment_ready": assert_phase9_broad_environment_ready,
            "build_phase9_broad_opponent_lanes": build_phase9_broad_opponent_lanes,
            "phase9_broad_opponent_mix": phase9_broad_opponent_mix,
            "prepare_phase9_broad_opponent_environment": prepare_phase9_broad_opponent_environment,
            "validate_broad_opponent_lanes": validate_broad_opponent_lanes,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "InfoModeV5",
    "AssistModeV5",
    "OBS_V5_DIM",
    "AssemblerCandidate",
    "AssemblerDatasetRow",
    "DesirererDatasetRow",
    "DeckMatchupEvaluator",
    "DrawDesirerer",
    "DrawAssistController",
    "DrawScore",
    "AUX_DATASET_SCHEMA",
    "LEAGUE_RUN_SCHEMA",
    "TRACE_POOL_SCHEMA",
    "AuxDatasetManifest",
    "LeagueRunManifest",
    "TracePoolEntry",
    "TracePoolManifest",
    "TrainV3ClassicEnv",
    "TrainV3EnvConfig",
    "V5EpisodeModes",
    "V5LeagueConfig",
    "V5RewardWeights",
    "V5TraceScenario",
    "build_golden_trace",
    "build_assembler_rows_from_matchup_summaries",
    "build_desirerer_rows_from_v5_trace",
    "compute_history_outcome_deltas_v5",
    "compute_reward_components_v5",
    "compute_weighted_reward_v5",
    "encode_observation_v5",
    "parse_v5_opponent_mix",
    "reward_snapshot_v5",
    "sample_v5_episode_modes",
    "RustBatchWorker",
    "RustCompactArgmaxActions",
    "RustDenseArgmaxActions",
    "RustPaddedArgmaxActions",
    "RustPaddedLegalActions",
    "RustPackedLegalRows",
    "RustPreparedPPOBatch",
    "CompactLegalActionScores",
    "PaddedLegalActionScores",
    "RustTransitionBatch",
    "RustPPOBatch",
    "RustPPOEvaluation",
    "RustPPOTrainingConfig",
    "RustRolloutStats",
    "RustTraceRolloutRunner",
    "RustVecEnv",
    "RustVecEnvReset",
    "RustVecEnvStep",
    "benchmark_compact_legal_policy_inference",
    "benchmark_rust_gae_prepare",
    "benchmark_rust_pre_step_action_tape_batch_modes",
    "benchmark_rust_ppo_update_modes",
    "benchmark_trainv3_speed_report",
    "benchmark_rust_vec_collector_modes",
    "benchmark_rust_vec_policy_collector_modes",
    "benchmark_trace_file",
    "compact_argmax_actions",
    "collect_rust_vec_rollout",
    "compare_adaptive_strength_monotonicity",
    "compute_rust_compact_argmax_actions",
    "compute_rust_dense_argmax_actions",
    "compute_rust_gae_returns",
    "compute_rust_pad_legal_actions",
    "compute_rust_pack_legal_action_rows",
    "compute_rust_padded_argmax_actions",
    "compute_rust_prepare_ppo_batch",
    "compute_rust_repeat_row_indices",
    "compute_rust_selected_local_indices",
    "evaluate_dense_rust_ppo_batch",
    "evaluate_adaptive_strength_proxy",
    "evaluate_assembler_baseline",
    "evaluate_desirerer_baseline",
    "evaluate_rust_ppo_batch",
    "generate_v5_trace_pool",
    "group_v5_trace_pool_by_mode",
    "load_v5_trace_pool_manifest",
    "load_assembler_dataset",
    "load_desirerer_dataset",
    "manifest_to_dict",
    "make_compact_legal_argmax_policy",
    "make_dense_argmax_policy",
    "make_padded_legal_argmax_policy",
    "padded_argmax_actions",
    "prepare_rust_ppo_batch",
    "read_manifest_json",
    "resolve_v5_trace_paths",
    "run_v5_adaptive_training_pipeline",
    "save_assembler_dataset",
    "save_assembler_dataset_with_manifest",
    "save_desirerer_dataset",
    "save_desirerer_dataset_with_manifest",
    "score_compact_legal_actions",
    "score_padded_legal_actions",
    "select_v5_trace_paths_for_mode",
    "train_dense_rust_ppo_minibatch",
    "train_v5_adaptive_main",
    "V5ActionConditionedPolicy",
    "create_v5_policy",
    "OpenAICompatibleTeacherClient",
    "OpenAICompatibleTeacherConfig",
    "TeacherPreferenceRow",
    "V5GauntletConfig",
    "ExploitLaneConfig",
    "build_default_exploit_gauntlet",
    "V5OpponentLane",
    "assert_phase9_broad_environment_ready",
    "build_phase9_broad_opponent_lanes",
    "phase9_broad_opponent_mix",
    "prepare_phase9_broad_opponent_environment",
    "validate_broad_opponent_lanes",
    "build_assembler_rows_from_v5_trace",
    "train_rust_ppo_trace_file",
    "train_rust_ppo_trace_files",
    "train_rust_ppo_minibatch",
    "transition_batch_from_action_tape_rollout",
    "write_manifest_json",
]
