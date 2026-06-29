use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::json;
use trainv3_core::exploit::{
    run_exploit_lane, ExploitAgentKind, ExploitLaneReport, LevelHandicap,
};
use trainv3_core::kernel::{
    compute_reward_components_v5, hash_f32_le, DrawRng, GoldenSnapshot, GoldenTrace, KernelConfig,
    KernelSnapshotOutput, RewardComponentsV5, RolloutKernel,
};
use trainv3_core::worker::{BatchedRolloutWorker, WorkerRng};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: trainv3_kernel <verify|bench|bench-step|bench-batch|gauntlet|gauntlet-manifest> <golden-trace.json|trace-manifest.json> [iterations|limit|steps] [env_count|steps|p1_level_delta] [p2_level_delta]"
        );
        std::process::exit(2);
    }

    let mode = args[1].as_str();
    match mode {
        "verify" => {
            let trace = read_trace(&args[2]);
            verify_trace(&trace);
        }
        "bench" => {
            let trace = read_trace(&args[2]);
            let iterations = args
                .get(3)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(10_000);
            bench_trace(&trace, iterations);
        }
        "bench-step" => {
            let trace = read_trace(&args[2]);
            let iterations = args
                .get(3)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(10_000);
            bench_step_trace(&trace, iterations);
        }
        "bench-batch" => {
            let trace = read_trace(&args[2]);
            let iterations = args
                .get(3)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1_000);
            let env_count = args
                .get(4)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64);
            bench_batch_trace(&trace, iterations, env_count);
        }
        "gauntlet" => {
            let trace = read_trace(&args[2]);
            let steps = args
                .get(3)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(32);
            let p1_delta = args.get(4).and_then(|v| v.parse::<i32>().ok()).unwrap_or(0);
            let p2_delta = args.get(5).and_then(|v| v.parse::<i32>().ok()).unwrap_or(0);
            gauntlet_trace(&trace, steps, p1_delta, p2_delta);
        }
        "gauntlet-manifest" => {
            let limit = args
                .get(3)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64);
            let steps = args
                .get(4)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(32);
            let p1_delta = args.get(5).and_then(|v| v.parse::<i32>().ok()).unwrap_or(0);
            let p2_delta = args.get(6).and_then(|v| v.parse::<i32>().ok()).unwrap_or(0);
            gauntlet_manifest(&args[2], limit, steps, p1_delta, p2_delta);
        }
        _ => {
            eprintln!("unknown mode: {mode}");
            std::process::exit(2);
        }
    }
}

fn gauntlet_trace(trace: &GoldenTrace, steps: usize, p1_delta: i32, p2_delta: i32) {
    println!("{}", gauntlet_trace_report(trace, steps, p1_delta, p2_delta));
}

fn gauntlet_trace_report(
    trace: &GoldenTrace,
    steps: usize,
    p1_delta: i32,
    p2_delta: i32,
) -> serde_json::Value {
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let handicap = LevelHandicap {
        p1_level_delta: p1_delta,
        p2_level_delta: p2_delta,
        min_level: 1,
        max_level: 10,
    };
    let kinds = [
        ExploitAgentKind::FaceRush,
        ExploitAgentKind::BoardControl,
        ExploitAgentKind::GreedyTrade,
        ExploitAgentKind::Stall,
        ExploitAgentKind::PunishEmptyBoard,
        ExploitAgentKind::AntiDrawGreed,
        ExploitAgentKind::AntiHandLeakOverfit,
    ];
    let started = Instant::now();
    let reports: Vec<ExploitLaneReport> = kinds
        .iter()
        .copied()
        .map(|kind| {
            run_exploit_lane(&trace.initial.state, config, kind, steps, handicap)
                .unwrap_or_else(|err| {
                    eprintln!("gauntlet lane {kind:?} failed: {err}");
                    std::process::exit(1);
                })
        })
        .collect();
    let elapsed = started.elapsed().as_secs_f64();
    let invalid_actions: usize = reports.iter().map(|report| report.invalid_actions).sum();
    let executed_steps: usize = reports.iter().map(|report| report.executed_steps).sum();
    let lanes: Vec<serde_json::Value> = reports
        .iter()
        .map(|report| {
            json!({
                "kind": format!("{:?}", report.kind),
                "requested_steps": report.requested_steps,
                "executed_steps": report.executed_steps,
                "invalid_actions": report.invalid_actions,
                "cumulative_reward": report.cumulative_reward,
                "terminal": report.terminal,
                "final_p1_hero_hp": report.final_p1_hero_hp,
                "final_p2_hero_hp": report.final_p2_hero_hp,
                "final_p1_board_count": report.final_p1_board_count,
                "final_p2_board_count": report.final_p2_board_count,
            })
        })
        .collect();
    json!({
        "ok": invalid_actions == 0,
        "schema": "trainv3-v5-rust-gauntlet-v1",
        "trace_schema": trace.schema,
        "steps_per_lane": steps,
        "p1_level_delta": p1_delta,
        "p2_level_delta": p2_delta,
        "lanes": lanes,
        "lane_count": reports.len(),
        "executed_steps": executed_steps,
        "invalid_actions": invalid_actions,
        "elapsed_ms": elapsed * 1000.0,
        "steps_per_second": executed_steps as f64 / elapsed.max(1e-9),
    })
}

fn read_trace(path: &str) -> GoldenTrace {
    let raw = fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("failed to read {path}: {err}");
        std::process::exit(2);
    });
    serde_json::from_str(&raw).unwrap_or_else(|err| {
        eprintln!("failed to parse {path}: {err}");
        std::process::exit(2);
    })
}

fn gauntlet_manifest(path: &str, limit: usize, steps: usize, p1_delta: i32, p2_delta: i32) {
    let manifest_path = Path::new(path);
    let raw = fs::read_to_string(manifest_path).unwrap_or_else(|err| {
        eprintln!("failed to read manifest {path}: {err}");
        std::process::exit(2);
    });
    let manifest: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|err| {
        eprintln!("failed to parse manifest {path}: {err}");
        std::process::exit(2);
    });
    let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let trace_paths = trace_paths_from_manifest(&manifest, base_dir);
    if trace_paths.is_empty() {
        eprintln!("manifest {path} has no trace paths");
        std::process::exit(2);
    }
    let selected = trace_paths.into_iter().take(limit.max(1)).collect::<Vec<_>>();
    let started = Instant::now();
    let mut trace_reports = Vec::with_capacity(selected.len());
    let mut total_invalid = 0_usize;
    let mut total_executed = 0_usize;
    for trace_path in &selected {
        let trace = read_trace(&trace_path.to_string_lossy());
        let report = gauntlet_trace_report(&trace, steps, p1_delta, p2_delta);
        total_invalid += report["invalid_actions"].as_u64().unwrap_or(0) as usize;
        total_executed += report["executed_steps"].as_u64().unwrap_or(0) as usize;
        trace_reports.push(json!({
            "trace_path": trace_path,
            "report": report,
        }));
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({
            "ok": total_invalid == 0,
            "schema": "trainv3-v5-rust-gauntlet-manifest-v1",
            "manifest_path": path,
            "trace_count": selected.len(),
            "steps_per_lane": steps,
            "p1_level_delta": p1_delta,
            "p2_level_delta": p2_delta,
            "invalid_actions": total_invalid,
            "executed_steps": total_executed,
            "elapsed_ms": elapsed * 1000.0,
            "steps_per_second": total_executed as f64 / elapsed.max(1e-9),
            "traces": trace_reports,
        })
    );
}

fn trace_paths_from_manifest(manifest: &serde_json::Value, base_dir: &Path) -> Vec<PathBuf> {
    let raw_paths = manifest
        .get("trace_paths")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_else(|| {
            manifest
                .get("traces")
                .and_then(|value| value.as_array())
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| entry.get("path").cloned())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        });
    raw_paths
        .iter()
        .filter_map(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            let path = PathBuf::from(value);
            if path.is_absolute() {
                path
            } else {
                base_dir.join(path)
            }
        })
        .collect()
}

fn verify_trace(trace: &GoldenTrace) {
    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));
    let mut snapshot_count = 0_usize;
    verify_snapshot(&kernel, "initial", &trace.initial);
    snapshot_count += 1;

    for step in &trace.steps {
        verify_snapshot(&kernel, &format!("step-{}-pre", step.t), &step.pre);
        verify_snapshot(&kernel, &format!("step-{}-post", step.t), &step.post);
        assert_reward_components_close(
            compute_reward_components_v5(&step.pre.state, &step.post.state, step.acting_player_id),
            step.reward_components_v5,
            &format!("step-{}-reward", step.t),
        );
        let mut rng = WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let transition = kernel
            .apply_action(&step.pre.state, step.acting_player_id, step.action_id, step.mana_draw_taken, &mut draw_rng)
            .unwrap_or_else(|err| {
                eprintln!("golden trace transition failed: step-{} {err}", step.t);
                std::process::exit(1);
            });
        let out = kernel.encode_snapshot_with_history(
            &transition.state,
            transition.state.current_turn_owner_id,
            &step.post.history_events,
        );
        verify_snapshot_output(&format!("step-{}-transition", step.t), &out, &step.post);
        assert_reward_components_close(
            transition.reward_components_v5,
            step.reward_components_v5,
            &format!("step-{}-transition-reward", step.t),
        );
        snapshot_count += 2;
    }

    println!(
        "{}",
        json!({
            "ok": true,
            "schema": trace.schema,
            "steps": trace.steps.len(),
            "snapshots": snapshot_count,
        })
    );
}

fn verify_snapshot(kernel: &RolloutKernel, label: &str, snapshot: &GoldenSnapshot) {
    let out = kernel.encode_snapshot_with_history(
        &snapshot.state,
        snapshot.state.current_turn_owner_id,
        &snapshot.history_events,
    );
    verify_snapshot_output(label, &out, snapshot);
}

fn verify_snapshot_output(label: &str, out: &KernelSnapshotOutput, snapshot: &GoldenSnapshot) {
    if out.legal_ids() != snapshot.legal_ids {
        fail(label, "legal_ids");
    }
    if hash_f32_le(&out.action_mask) != snapshot.mask_sha256_f32_le {
        fail(label, "action_mask");
    }
    if hash_f32_le(&out.action_features) != snapshot.action_features_sha256_f32_le {
        fail(label, "action_features");
    }
    if hash_f32_le(&out.observation_v1) != snapshot.obs_sha256_f32_le {
        fail(label, "observation_v1");
    }
    if let Some(expected) = snapshot.obs_v5_sha256_f32_le.as_deref() {
        if hash_f32_le(&out.observation_v5) != expected {
            fail(label, "observation_v5");
        }
    }
}

fn bench_trace(trace: &GoldenTrace, iterations: usize) {
    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));
    let mut snapshots: Vec<&GoldenSnapshot> = Vec::with_capacity(1 + trace.steps.len() * 2);
    snapshots.push(&trace.initial);
    for step in &trace.steps {
        snapshots.push(&step.pre);
        snapshots.push(&step.post);
    }

    let started = Instant::now();
    let mut checksum = 0.0_f32;
    for _ in 0..iterations {
        for snapshot in &snapshots {
            let out = kernel.encode_snapshot_with_history(
                &snapshot.state,
                snapshot.state.current_turn_owner_id,
                &snapshot.history_events,
            );
            checksum += out.action_mask[0] + out.observation_v1[0] + out.observation_v5[0];
        }
    }
    let elapsed = started.elapsed();
    let encoded = iterations * snapshots.len();
    let elapsed_s = elapsed.as_secs_f64();
    println!(
        "{}",
        json!({
            "ok": true,
            "snapshots_per_iteration": snapshots.len(),
            "iterations": iterations,
            "encoded_snapshots": encoded,
            "elapsed_ms": elapsed_s * 1000.0,
            "snapshots_per_second": encoded as f64 / elapsed_s,
            "microseconds_per_snapshot": elapsed_s * 1_000_000.0 / encoded as f64,
            "checksum": checksum,
        })
    );
}

fn bench_step_trace(trace: &GoldenTrace, iterations: usize) {
    let kernel = RolloutKernel::new(KernelConfig::from_trace_config(&trace.env_config));
    let started = Instant::now();
    let mut checksum = 0.0_f32;
    for _ in 0..iterations {
        for step in &trace.steps {
            let mut rng = WorkerRng::Deterministic;
            let mut draw_rng = DrawRng::live(&mut rng);
            let transition = kernel
                .apply_action(&step.pre.state, step.acting_player_id, step.action_id, step.mana_draw_taken, &mut draw_rng)
                .expect("fixture action applies");
            let out = kernel.encode_snapshot_with_history(
                &transition.state,
                transition.state.current_turn_owner_id,
                &step.post.history_events,
            );
            checksum += out.action_mask[0]
                + out.observation_v1[0]
                + out.observation_v5[0]
                + transition.reward_components_v5.hp_potential_delta;
        }
    }
    let elapsed = started.elapsed();
    let transitions = iterations * trace.steps.len();
    let elapsed_s = elapsed.as_secs_f64();
    println!(
        "{}",
        json!({
            "ok": true,
            "steps_per_iteration": trace.steps.len(),
            "iterations": iterations,
            "transitions": transitions,
            "elapsed_ms": elapsed_s * 1000.0,
            "transitions_per_second": transitions as f64 / elapsed_s,
            "microseconds_per_transition": elapsed_s * 1_000_000.0 / transitions as f64,
            "checksum": checksum,
        })
    );
}

fn bench_batch_trace(trace: &GoldenTrace, iterations: usize, env_count: usize) {
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let action_ids: Vec<usize> = trace.steps.iter().map(|step| step.action_id).collect();
    let snapshots = vec![trace.initial.clone(); env_count];
    let mut worker = BatchedRolloutWorker::from_snapshots(config, &snapshots);
    worker.use_deterministic_rng();
    let mut step_actions = vec![0_usize; env_count];

    let started = Instant::now();
    let mut checksum = 0.0_f32;
    for _ in 0..iterations {
        worker.reset_all();
        for action_id in &action_ids {
            step_actions.fill(*action_id);
            let out = worker.step(&step_actions).expect("fixture batch applies");
            checksum += out.rewards.iter().sum::<f32>();
            checksum += out.observation_v1.first().copied().unwrap_or(0.0);
            checksum += out.observation_v5.first().copied().unwrap_or(0.0);
        }
    }
    let elapsed = started.elapsed();
    let env_transitions = iterations * trace.steps.len() * env_count;
    let elapsed_s = elapsed.as_secs_f64();
    println!(
        "{}",
        json!({
            "ok": true,
            "env_count": env_count,
            "steps_per_iteration": trace.steps.len(),
            "iterations": iterations,
            "worker_reuse": true,
            "reset_per_iteration": true,
            "env_transitions": env_transitions,
            "elapsed_ms": elapsed_s * 1000.0,
            "env_transitions_per_second": env_transitions as f64 / elapsed_s,
            "microseconds_per_env_transition": elapsed_s * 1_000_000.0 / env_transitions as f64,
            "checksum": checksum,
        })
    );
}

fn assert_reward_components_close(
    actual: RewardComponentsV5,
    expected: RewardComponentsV5,
    label: &str,
) {
    let close = (actual.hp_potential_delta - expected.hp_potential_delta).abs() < 1e-6
        && (actual.board_power_delta - expected.board_power_delta).abs() < 1e-6
        && (actual.my_board_power - expected.my_board_power).abs() < 1e-6
        && (actual.enemy_board_power - expected.enemy_board_power).abs() < 1e-6
        && (actual.board_power_ratio - expected.board_power_ratio).abs() < 1e-6
        && actual.board_under_0_7 == expected.board_under_0_7
        && actual.own_board_wiped == expected.own_board_wiped
        && actual.my_board_count_delta == expected.my_board_count_delta
        && actual.enemy_board_count_delta == expected.enemy_board_count_delta;
    if !close {
        fail(label, "reward_components_v5");
    }
}

fn fail(label: &str, field: &str) -> ! {
    eprintln!("golden trace mismatch: {label} {field}");
    std::process::exit(1);
}
