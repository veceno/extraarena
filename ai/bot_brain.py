"""
Инференс обученных ONNX-моделей для боевых ботов с поддержкой профилей сложности.
Преобразует GameState -> action_id с маскированием и температурным сэмплированием.

Интеграция:
    1. Модели загружаются при старте сервера (web/server.py create_web_app()).
    2. Бот-матчи используют ONNX при наличии загруженной модели (engine.is_bot = True).
    3. Поддержка моделей: TrainV2 v4 ONNX (`train_v2_classic_v1`)
    4. Температурный сэмплинг: T ∈ [0.1, 1.8] для контроля случайности
    5. Маскирование: недопустимые действия получают логит -1e9

Поддержка форматов наблюдений:
    train_v2_classic_v1: obs_dim=1456, action_feature_dim=171,
    max_candidate_actions=601.

Зависимости:
    - onnxruntime (установить: pip install onnxruntime)
    - numpy (уже в проекте)
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from core.actions import BaseAction
from core.state import GameState

logger = logging.getLogger(__name__)

_TRAIN_V2_FORMAT = "train_v2_classic_v1"
_V5_FORMAT = "v5"


class BerserkInference:
    """
    ONNX-инференс бота с поддержкой множественных профилей сложности.
    Управляет словарём сессий для разных моделей и применяет температурный сэмплинг.
    """

    @staticmethod
    def _resolve_model_path(raw_model_path: str | Path) -> Path:
        model_path = Path(raw_model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).parent.parent / model_path
        return model_path

    @staticmethod
    def _load_sidecar(model_path: Path) -> dict[str, Any]:
        sidecar_path = Path(str(model_path) + ".json")
        if not sidecar_path.exists():
            return {}
        try:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            config = data.get("config")
            if isinstance(config, dict):
                for key in (
                    "include_preview_features",
                    "verify_mask",
                    "placement_mode",
                    "action_codec",
                    "observation_codec",
                ):
                    if key not in data and key in config:
                        data[key] = config[key]
            return data
        except Exception as exc:
            logger.warning("[BerserkInference] Failed to read sidecar %s: %s", sidecar_path, exc)
            return {}

    @staticmethod
    def _validate_temperature_range(difficulty: str, profile: dict[str, Any]) -> tuple[float, float]:
        if "temperature_range" not in profile:
            raise ValueError(f"{difficulty}: missing temperature_range")
        raw = profile["temperature_range"]
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError(f"{difficulty}: temperature_range must contain two numbers")
        temp_min = float(raw[0])
        temp_max = float(raw[1])
        if temp_min <= 0 or temp_max <= 0 or temp_min > temp_max:
            raise ValueError(f"{difficulty}: invalid temperature_range={raw}")
        return temp_min, temp_max

    @staticmethod
    def _validate_selection(difficulty: str, profile: dict[str, Any]) -> str:
        selection = str(profile.get("selection", "argmax"))
        if selection not in {"argmax", "softmax"}:
            raise ValueError(f"{difficulty}: invalid selection={selection!r}")
        return selection

    @staticmethod
    def _validate_placement_mode(difficulty: str, profile: dict[str, Any]) -> str:
        placement_mode = str(profile.get("placement_mode", "append_only"))
        if placement_mode not in {"append_only", "full"}:
            raise ValueError(f"{difficulty}: invalid placement_mode={placement_mode!r}")
        return placement_mode

    @staticmethod
    def _validate_verify_mask(difficulty: str, profile: dict[str, Any]) -> bool:
        verify_mask = profile.get("verify_mask", True)
        if not isinstance(verify_mask, bool):
            raise ValueError(f"{difficulty}: verify_mask must be boolean, got {verify_mask!r}")
        return verify_mask

    @staticmethod
    def _validate_codec_options(difficulty: str, profile: dict[str, Any]) -> tuple[str, str]:
        action_codec = str(profile.get("action_codec", "classic_actions_v1"))
        observation_codec = str(profile.get("observation_codec", "classic_obs_v1"))
        if action_codec != "classic_actions_v1":
            raise ValueError(f"{difficulty}: invalid action_codec={action_codec!r}")
        if observation_codec != "classic_obs_v1":
            raise ValueError(f"{difficulty}: invalid observation_codec={observation_codec!r}")
        return action_codec, observation_codec

    @staticmethod
    def _shape_dim(shape: list[Any] | tuple[Any, ...] | None, index: int) -> int | None:
        if not shape or len(shape) <= index:
            return None
        value = shape[index]
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _validate_train_v2_contract(
        cls,
        difficulty: str,
        session: ort.InferenceSession,
        profile: dict[str, Any],
    ) -> tuple[list[str], list[str], int, int, bool]:
        obs_dim = int(profile.get("obs_dim", 0))
        action_feature_dim = int(profile.get("action_feature_dim", 0))
        max_candidate_actions = int(profile.get("max_candidate_actions", 0))

        if obs_dim != 1456:
            raise ValueError(f"{difficulty}: train_v2 obs_dim must be 1456, got {obs_dim}")
        if action_feature_dim != 171:
            raise ValueError(f"{difficulty}: action_feature_dim must be 171, got {action_feature_dim}")
        if max_candidate_actions != 601:
            raise ValueError(f"{difficulty}: max_candidate_actions must be 601, got {max_candidate_actions}")

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        input_names = [i.name for i in inputs]
        output_names = [o.name for o in outputs]
        if "observation" not in input_names or "action_features" not in input_names:
            raise ValueError(f"{difficulty}: train_v2 inputs must include observation/action_features, got {input_names}")
        if "logits" not in output_names:
            raise ValueError(f"{difficulty}: train_v2 outputs must include logits, got {output_names}")

        meta_by_name = {i.name: i for i in inputs}
        obs_meta = meta_by_name["observation"]
        af_meta = meta_by_name["action_features"]
        obs_shape_dim = cls._shape_dim(obs_meta.shape, 1)
        if obs_shape_dim is not None and obs_shape_dim != obs_dim:
            raise ValueError(f"{difficulty}: observation input shape {obs_meta.shape} != obs_dim {obs_dim}")
        af_actions = cls._shape_dim(af_meta.shape, 1)
        af_dim = cls._shape_dim(af_meta.shape, 2)
        if af_actions is not None and af_actions != max_candidate_actions:
            raise ValueError(
                f"{difficulty}: action_features candidate shape {af_meta.shape} != {max_candidate_actions}"
            )
        if af_dim is not None and af_dim != action_feature_dim:
            raise ValueError(
                f"{difficulty}: action_features feature shape {af_meta.shape} != {action_feature_dim}"
            )

        output_by_name = {o.name: o for o in outputs}
        logits_meta = output_by_name["logits"]
        logits_actions = cls._shape_dim(logits_meta.shape, 1)
        if logits_actions is not None and logits_actions != max_candidate_actions:
            raise ValueError(f"{difficulty}: logits output shape {logits_meta.shape} != {max_candidate_actions}")

        include_preview = bool(profile.get("include_preview_features", False))
        return input_names, output_names, action_feature_dim, max_candidate_actions, include_preview

    @classmethod
    def _validate_v5_contract(
        cls,
        difficulty: str,
        session: ort.InferenceSession,
        profile: dict[str, Any],
    ) -> tuple[list[str], list[str], int, int, bool]:
        """Validate the V5 ONNX contract (3-output: logits + value + mana_draw_logit).

        Mirrors ``_validate_train_v2_contract`` but asserts the V5 shape:
        obs_dim==7128, action_feature_dim==171, max_candidate_actions==601,
        inputs include observation+action_features, outputs include
        logits+value+mana_draw_logit (the 3-tuple), and mana_draw_head is
        truthy in the profile. Does NOT touch _validate_train_v2_contract (V4
        stays byte-unchanged).
        """
        obs_dim = int(profile.get("obs_dim", 0))
        action_feature_dim = int(profile.get("action_feature_dim", 0))
        max_candidate_actions = int(profile.get("max_candidate_actions", 0))

        if obs_dim != 7128:
            raise ValueError(f"{difficulty}: v5 obs_dim must be 7128, got {obs_dim}")
        if action_feature_dim != 171:
            raise ValueError(f"{difficulty}: v5 action_feature_dim must be 171, got {action_feature_dim}")
        if max_candidate_actions != 601:
            raise ValueError(f"{difficulty}: v5 max_candidate_actions must be 601, got {max_candidate_actions}")

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        input_names = [i.name for i in inputs]
        output_names = [o.name for o in outputs]
        if "observation" not in input_names or "action_features" not in input_names:
            raise ValueError(f"{difficulty}: v5 inputs must include observation/action_features, got {input_names}")
        if "logits" not in output_names or "value" not in output_names or "mana_draw_logit" not in output_names:
            raise ValueError(
                f"{difficulty}: v5 outputs must include logits+value+mana_draw_logit (3-tuple), got {output_names}"
            )

        meta_by_name = {i.name: i for i in inputs}
        obs_meta = meta_by_name["observation"]
        af_meta = meta_by_name["action_features"]
        obs_shape_dim = cls._shape_dim(obs_meta.shape, 1)
        if obs_shape_dim is not None and obs_shape_dim != obs_dim:
            raise ValueError(f"{difficulty}: v5 observation input shape {obs_meta.shape} != obs_dim {obs_dim}")
        af_actions = cls._shape_dim(af_meta.shape, 1)
        af_dim = cls._shape_dim(af_meta.shape, 2)
        if af_actions is not None and af_actions != max_candidate_actions:
            raise ValueError(
                f"{difficulty}: v5 action_features candidate shape {af_meta.shape} != {max_candidate_actions}"
            )
        if af_dim is not None and af_dim != action_feature_dim:
            raise ValueError(
                f"{difficulty}: v5 action_features feature shape {af_meta.shape} != {action_feature_dim}"
            )

        output_by_name = {o.name: o for o in outputs}
        logits_meta = output_by_name["logits"]
        logits_actions = cls._shape_dim(logits_meta.shape, 1)
        if logits_actions is not None and logits_actions != max_candidate_actions:
            raise ValueError(f"{difficulty}: v5 logits output shape {logits_meta.shape} != {max_candidate_actions}")

        if not profile.get("mana_draw_head"):
            raise ValueError(f"{difficulty}: v5 mana_draw_head must be truthy in the profile")

        include_preview = bool(profile.get("include_preview_features", False))
        return input_names, output_names, action_feature_dim, max_candidate_actions, include_preview

    def __init__(
        self,
        profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Args:
            profiles: Словарь профилей вида {difficulty: {model_path, obs_dim, temperature_range}}
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        if profiles is None:
            profiles = {}
        
        # Загружаем все профили. Несколько difficulty/tier keys могут ссылаться
        # на один ONNX-файл с разной temperature/selection, поэтому сессии
        # переиспользуются по пути модели.
        session_cache: Dict[str, ort.InferenceSession] = {}
        load_errors: list[tuple[str, Exception]] = []
        for difficulty, profile in profiles.items():
            try:
                model_path = self._resolve_model_path(profile["model_path"])

                if not model_path.exists():
                    exc = FileNotFoundError(f"{difficulty}: model not found: {model_path}")
                    logger.error("[BerserkInference] %s", exc)
                    load_errors.append((difficulty, exc))
                    continue

                sidecar = self._load_sidecar(model_path)
                merged_profile = dict(sidecar)
                merged_profile.update(profile)
                profile_format = merged_profile.get("format", None)
                if profile_format not in (_TRAIN_V2_FORMAT, _V5_FORMAT):
                    logger.warning(
                        "[BerserkInference] %s skipped: legacy/non-TrainV2 profile format=%s",
                        difficulty,
                        profile_format,
                    )
                    continue

                temp_range = self._validate_temperature_range(difficulty, merged_profile)
                merged_profile["temperature_range"] = temp_range
                selection = self._validate_selection(difficulty, merged_profile)
                placement_mode = self._validate_placement_mode(difficulty, merged_profile)
                verify_mask = self._validate_verify_mask(difficulty, merged_profile)
                action_codec, observation_codec = self._validate_codec_options(difficulty, merged_profile)

                cache_key = str(model_path.resolve())
                session = session_cache.get(cache_key)
                if session is None:
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=["CPUExecutionProvider"],
                    )
                    session_cache[cache_key] = session

                if profile_format == _V5_FORMAT:
                    # V5 branch -- validate the V5 3-output contract (logits +
                    # value + mana_draw_logit) and store the session with
                    # format="v5" + mana_draw_head. The V4 path
                    # (_validate_train_v2_contract + the V4 session dict) is
                    # byte-unchanged in the else branch below.
                    (
                        input_names,
                        output_names,
                        af_dim,
                        max_acts,
                        include_preview,
                    ) = self._validate_v5_contract(difficulty, session, merged_profile)

                    self.sessions[difficulty] = {
                        "session": session,
                        "format": profile_format,
                        "obs_dim": int(merged_profile["obs_dim"]),
                        "action_feature_dim": af_dim,
                        "max_candidate_actions": max_acts,
                        "include_preview_features": include_preview,
                        "temperature_range": temp_range,
                        "selection": selection,
                        "placement_mode": placement_mode,
                        "verify_mask": verify_mask,
                        "action_codec": action_codec,
                        "observation_codec": observation_codec,
                        "input_names": input_names,
                        "output_names": output_names,
                        "mana_draw_head": bool(merged_profile.get("mana_draw_head", True)),
                    }
                else:
                    (
                        input_names,
                        output_names,
                        af_dim,
                        max_acts,
                        include_preview,
                    ) = self._validate_train_v2_contract(difficulty, session, merged_profile)

                    self.sessions[difficulty] = {
                        "session": session,
                        "format": profile_format,
                        "obs_dim": int(merged_profile["obs_dim"]),
                        "action_feature_dim": af_dim,
                        "max_candidate_actions": max_acts,
                        "include_preview_features": include_preview,
                        "temperature_range": temp_range,
                        "selection": selection,
                        "placement_mode": placement_mode,
                        "verify_mask": verify_mask,
                        "action_codec": action_codec,
                        "observation_codec": observation_codec,
                        "input_names": input_names,
                        "output_names": output_names,
                    }

                input_meta = session.get_inputs()[0]
                output_meta = session.get_outputs()[0]
                logger.info(
                    f"[BerserkInference] {difficulty}: {model_path.name}, "
                    f"input={input_meta.shape}, output={output_meta.shape}, "
                    f"T={temp_range}, sel={merged_profile.get('selection', 'softmax')}"
                    + (f", format={profile_format}" if profile_format else "")
                )
            except Exception as exc:
                logger.error(
                    f"[BerserkInference] Ошибка загрузки {difficulty}: {exc}",
                    exc_info=True,
                )
                load_errors.append((difficulty, exc))
        
        if load_errors:
            details = "; ".join(f"{name}: {exc}" for name, exc in load_errors)
            logger.error("[BerserkInference] Ошибка загрузки профилей: %s", details)
        if not self.sessions:
            logger.warning("[BerserkInference] Ни одна TrainV2 v4 модель не загружена; бот уйдет в rule-based fallback")

    def has_profile(self, difficulty: str) -> bool:
        """Return whether a difficulty profile is loaded and ready for inference."""
        return str(difficulty) in self.sessions

    def get_action(
        self,
        game_state: GameState,
        player_id: int,
        legal_actions: List[BaseAction],
        difficulty: str = "medium",
    ) -> int:
        """
        Получить индекс действия с температурным сэмплированием.

        Args:
            game_state: Текущее игровое состояние
            player_id: ID бота (владелец хода)
            legal_actions: Список легальных действий из engine.get_legal_actions()
            difficulty: Профиль сложности (lite/easy/medium/hard/max)

        Returns:
            Индекс действия в списке legal_actions
        """
        if not legal_actions:
            logger.warning("[BerserkInference] Нет доступных действий, возврат 0")
            return 0
        
        if difficulty not in self.sessions:
            raise ValueError(f"Unknown Berserk difficulty: {difficulty}")
        
        profile = self.sessions[difficulty]

        # TrainV2 branch — stable action_id → index in legal_actions
        if profile.get("format") == "train_v2_classic_v1":
            return self._get_action_train_v2_classic(
                game_state, player_id, legal_actions, difficulty, profile
            )

        if profile.get("format") == "v5":
            return self._get_action_v5(
                game_state, player_id, legal_actions, difficulty, profile
            )

        raise ValueError(f"Unsupported Berserk profile format: {profile.get('format')}")

    async def get_action_async(
        self,
        game_state: GameState,
        player_id: int,
        legal_actions: List[BaseAction],
        difficulty: str = "medium",
    ) -> int:
        """Run ONNX inference away from the event loop."""
        return await asyncio.to_thread(
            self.get_action,
            game_state,
            player_id,
            legal_actions,
            difficulty,
        )

    def close(self) -> None:
        """Release session references; useful for future hot-reload paths."""
        self.sessions.clear()

    def __enter__(self) -> "BerserkInference":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _find_matching_legal_action_index(
        decoded: BaseAction,
        legal_actions: List[BaseAction],
    ) -> int | None:
        # TODO: когда боевой runtime начнёт уважать placement, position для
        # PlayCardAction нужно будет сравнивать, а не игнорировать.
        d_dict = decoded.to_dict()
        d_type = d_dict.get("type")

        for i, legal in enumerate(legal_actions):
            l_dict = legal.to_dict()
            if d_dict == l_dict:
                return i

            if d_type == "end_turn" and l_dict.get("type") == "end_turn":
                return i

            if d_type == "play_card" and l_dict.get("type") == "play_card":
                if d_dict.get("hand_index") == l_dict.get("hand_index") and d_dict.get("target_id") == l_dict.get("target_id"):
                    return i

            if d_type == "attack" and l_dict.get("type") == "attack":
                if d_dict.get("attacker_id") == l_dict.get("attacker_id") and d_dict.get("target_is_hero") == l_dict.get("target_is_hero") and d_dict.get("target_id") == l_dict.get("target_id"):
                    return i

        return None

    @staticmethod
    def _resolve_train_v2_io_names(profile: dict) -> tuple[str, str, list[str] | None]:
        input_names = profile.get("input_names", [])
        output_names = profile.get("output_names", [])

        if len(input_names) < 2:
            logger.warning(
                "[BerserkInference] TrainV2 profile has < 2 input_names: %s, fallback",
                input_names,
            )
            return "", "", None

        obs_name = "observation" if "observation" in input_names else input_names[0]
        af_name = "action_features" if "action_features" in input_names else input_names[1]

        if "logits" in output_names and "value" in output_names:
            outputs = ["logits", "value"]
        elif "logits" in output_names:
            outputs = ["logits"]
        else:
            outputs = output_names[:2] if len(output_names) >= 2 else None

        return obs_name, af_name, outputs

    def _get_action_train_v2_classic(
        self,
        game_state: GameState,
        player_id: int,
        legal_actions: List[BaseAction],
        difficulty: str,
        profile: dict,
    ) -> int:
        if not legal_actions:
            logger.warning("[BerserkInference] Нет доступных действий, возврат 0")
            return 0

        from ai.train_v2.classic_obs_v1 import encode_observation
        from ai.train_v2.classic_actions_v1 import (
            build_action_mask,
            encode_action_features,
            decode_action,
        )

        session = profile["session"]
        temp_min, temp_max = profile["temperature_range"]
        selection_mode = profile.get("selection", "argmax")

        obs_dim = int(profile.get("obs_dim", 1456))
        max_candidate_actions = int(profile.get("max_candidate_actions", 601))
        action_feature_dim = int(profile.get("action_feature_dim", 171))

        obs = encode_observation(game_state, player_id).reshape(1, obs_dim).astype(np.float32)
        placement_mode = profile.get("placement_mode", "append_only")
        verify_mask = bool(profile.get("verify_mask", True))
        include_preview = bool(profile.get("include_preview_features", False))
        mask = build_action_mask(
            game_state,
            player_id,
            verify_mask=verify_mask,
            placement_mode=placement_mode,
        ).astype(np.float32)
        af = encode_action_features(
            game_state,
            player_id,
            include_preview=include_preview,
            verify_mask=verify_mask,
            placement_mode=placement_mode,
            mask=mask,
        ).reshape(1, max_candidate_actions, action_feature_dim).astype(np.float32)

        obs_name, af_name, output_names = self._resolve_train_v2_io_names(profile)
        if output_names is None:
            logger.warning(
                "[BerserkInference] TrainV2 profile has no valid output names, fallback"
            )
            return _legal_fallback(legal_actions)

        input_feed = {obs_name: obs, af_name: af}
        outputs = session.run(output_names, input_feed)
        logits = outputs[0][0]
        if logits.shape[0] < max_candidate_actions:
            logger.warning(
                "[BerserkInference] TrainV2 logits too short: got=%s expected=%s, fallback",
                logits.shape[0],
                max_candidate_actions,
            )
            return _legal_fallback(legal_actions)
        logits = logits[:max_candidate_actions]
        if not np.all(np.isfinite(logits)):
            logger.warning("[BerserkInference] TrainV2 logits contain non-finite values, fallback")
            return _legal_fallback(legal_actions)

        mlogits = np.where(mask.astype(bool), logits, -1e9).astype(np.float32)
        legal_mask = mask.astype(bool)
        if not np.any(legal_mask):
            logger.warning("[BerserkInference] TrainV2 mask has no legal actions, fallback")
            return _legal_fallback(legal_actions)

        if selection_mode == "argmax":
            action_id = int(np.argmax(mlogits))
        else:
            temperature = random.uniform(temp_min, temp_max)
            scaled = mlogits / temperature
            legal_scaled = scaled[legal_mask]
            legal_max = np.max(legal_scaled)
            if not np.isfinite(legal_max):
                logger.warning("[BerserkInference] TrainV2 softmax max invalid, fallback")
                return _legal_fallback(legal_actions)
            scaled = scaled - legal_max
            exps = np.zeros(max_candidate_actions, dtype=np.float32)
            exps[legal_mask] = np.exp(scaled[legal_mask])
            prob_sum = float(np.sum(exps))
            if prob_sum <= 1e-10 or not np.isfinite(prob_sum):
                logger.warning("[BerserkInference] TrainV2 softmax sum invalid: %s, fallback", prob_sum)
                return _legal_fallback(legal_actions)
            probs = exps / (prob_sum + 1e-10)
            action_id = int(np.random.choice(max_candidate_actions, p=probs))

        decoded = decode_action(game_state, player_id, action_id)
        if decoded is None:
            logger.warning(
                f"[BerserkInference] TrainV2 action_id={action_id} decode_action returned None, fallback"
            )
            return _legal_fallback(legal_actions)

        idx = self._find_matching_legal_action_index(decoded, legal_actions)
        if idx is None:
            logger.warning(
                f"[BerserkInference] TrainV2 action_id={action_id} decoded={decoded.to_dict()} not found in legal_actions, fallback"
            )
            return _legal_fallback(legal_actions)

        logger.debug(
            f"[BerserkInference] player={player_id}, difficulty={difficulty}, "
            f"selection={selection_mode}, legal={len(legal_actions)}, "
            f"train_v2_action={action_id} -> legal_idx={idx}"
        )

        return idx

    def _get_action_v5(
        self,
        game_state: GameState,
        player_id: int,
        legal_actions: List[BaseAction],
        difficulty: str,
        profile: dict,
    ) -> int:
        """V5 ONNX inference path -- 3-output contract + mana_draw head wiring.

        Mirrors ``_get_action_train_v2_classic`` but V5-shaped: encodes the
        7128-dim V5 observation, runs the 3-output ONNX (logits + value +
        mana_draw_logit), applies the ONNX fallback guard (last-resort prod
        safety -- raises RuntimeError on NaN/inf OR no-legal-candidate, NOT a
        silent _legal_fallback), then wires the mana_draw parallel binary head.

        mana_draw decision: ``mana_draw`` is a PARALLEL BINARY GATE, NOT a
        602nd candidate. When its legal-gated sigmoid probability exceeds 0.5,
        the method returns the ``ManaDrawAction`` index in ``legal_actions``.
        Otherwise it decodes the conditional 601-best candidate and matches it
        against ``legal_actions``.

        Exception handling: any UNEXPECTED exception (encoder/decoder/match
        failure) degrades to ``_legal_fallback`` (the V4 silent-fallback
        behavior). The ONNX fallback guard ``RuntimeError`` is NOT swallowed
        -- it MUST propagate out of ``_get_action_v5`` (a malformed V5 ONNX is
        the last-resort prod safety, NOT a silent rule-based fallback).
        """
        if not legal_actions:
            logger.warning("[BerserkInference] V5: нет доступных действий, возврат 0")
            return 0

        # Lazy-import the V5 encoder set + the V4 codec helpers + ManaDrawAction
        # so prod does NOT pay the import cost when only V4 is loaded (mirrors
        # the V4 lazy-import pattern). These import ONLY ai.train_v2.* (the
        # vendored V5 live-path copies) + core.* -- ZERO train_v3 / rlhf_env
        # imports on the live hot path.
        from ai.train_v2.obs_v5 import encode_observation_v5
        from ai.train_v2.v5_contracts import InfoModeV5
        from ai.train_v2.mana_draw_head_v5 import mana_draw_legal_mask, select_includes_mana_draw
        from ai.train_v2.classic_actions_v1 import (
            build_action_mask,
            encode_action_features,
            decode_action,
        )
        from ai.train_v2.v5_inference_guard import _assert_v5_logits_finite_legal
        from core.actions import ManaDrawAction

        session = profile["session"]
        obs_dim = int(profile.get("obs_dim", 7128))
        max_candidate_actions = int(profile.get("max_candidate_actions", 601))
        action_feature_dim = int(profile.get("action_feature_dim", 171))
        placement_mode = profile.get("placement_mode", "append_only")
        verify_mask = bool(profile.get("verify_mask", False))

        try:
            obs = encode_observation_v5(
                game_state,
                player_id,
                info_mode=InfoModeV5(enemy_hand_known=True, enemy_deck_known=True),
                history_events=list(getattr(game_state, "v5_history_events", ())),
            ).reshape(1, obs_dim).astype(np.float32)
            mask = build_action_mask(
                game_state,
                player_id,
                verify_mask=verify_mask,
                placement_mode=placement_mode,
            ).astype(np.float32)
            af = encode_action_features(
                game_state,
                player_id,
                include_preview=False,
                verify_mask=verify_mask,
                placement_mode=placement_mode,
                mask=mask,
            ).reshape(1, max_candidate_actions, action_feature_dim).astype(np.float32)

            obs_name, af_name, output_names = self._resolve_train_v2_io_names(profile)
            if output_names is None or "mana_draw_logit" not in output_names:
                # V5 contract requires the 3-tuple; fall back to the explicit
                # output list if _resolve_train_v2_io_names dropped mana_draw_logit
                # (it only knows the V4 2-tuple logits+value).
                obs_name = "observation" if not obs_name or "observation" not in (output_names or []) else obs_name
                af_name = "action_features" if not af_name else af_name
                output_names = ["logits", "value", "mana_draw_logit"]

            outputs = session.run(output_names, {"observation": obs, "action_features": af})
            logits = outputs[0][0]  # shape [601]
            value = outputs[1]  # noqa: F841 -- V5 value head (unused for action selection)
            mana_draw_logit = float(outputs[2][0][0])  # scalar

            # ONNX FALLBACK GUARD (SPEC :174) -- last-resort prod safety.
            # Raises RuntimeError on NaN/inf logits OR no-legal-candidate.
            # This RuntimeError MUST propagate (NOT swallowed into _legal_fallback).
            action_id = _assert_v5_logits_finite_legal(logits, mask)

            # Factorized decision: mana_draw is a legal-gated binary policy,
            # not a scalar to compare with an individual card logit.
            mana_draw_legal = mana_draw_legal_mask(game_state, player_id)
            candidate_placeholder_logit = float(logits[action_id])
            if select_includes_mana_draw(
                mana_draw_logit, candidate_placeholder_logit, mana_draw_legal
            ):
                for i, action in enumerate(legal_actions):
                    if isinstance(action, ManaDrawAction):
                        logger.debug(
                            f"[BerserkInference] V5 player={player_id}, difficulty={difficulty}, "
                            f"mana_draw_gate={mana_draw_logit:.3f} > 0 -> mana_draw legal_idx={i}"
                        )
                        return i
                # select_includes_mana_draw said legal but no ManaDrawAction in
                # legal_actions -- engine/encoder mismatch; fall through to the
                # 601-candidate decode (defensive).
                logger.warning(
                    "[BerserkInference] V5: select_includes_mana_draw fired but no "
                    "ManaDrawAction in legal_actions; falling back to 601-candidate"
                )

            decoded = decode_action(game_state, player_id, action_id)
            if decoded is None:
                logger.warning(
                    f"[BerserkInference] V5 action_id={action_id} decode_action returned None, fallback"
                )
                return _legal_fallback(legal_actions)

            idx = self._find_matching_legal_action_index(decoded, legal_actions)
            if idx is None:
                logger.warning(
                    f"[BerserkInference] V5 action_id={action_id} decoded={decoded.to_dict()} not found in legal_actions, fallback"
                )
                return _legal_fallback(legal_actions)

            logger.debug(
                f"[BerserkInference] V5 player={player_id}, difficulty={difficulty}, "
                f"legal={len(legal_actions)}, v5_action={action_id} -> legal_idx={idx}, "
                f"mana_draw_logit={mana_draw_logit:.3f}, mana_draw_legal={mana_draw_legal}"
            )
            return idx
        except RuntimeError:
            # The ONNX fallback guard RuntimeError MUST propagate -- a malformed
            # V5 ONNX is the last-resort prod safety, NOT a silent _legal_fallback.
            raise
        except Exception as exc:
            # Any OTHER unexpected exception (encoder/decoder/match failure)
            # degrades to _legal_fallback (mirrors the V4 silent-fallback path).
            logger.warning(
                "[BerserkInference] V5 unexpected exception: %s, fallback", exc, exc_info=True
            )
            return _legal_fallback(legal_actions)

def _legal_fallback(legal_actions: list) -> int:
    for i, action in enumerate(legal_actions):
        action_dict = action.to_dict()
        if action_dict.get("type") == "attack" and action_dict.get("target_is_hero") is True:
            return i
    for i, action in enumerate(legal_actions):
        if action.to_dict().get("type") == "attack":
            return i
    for i, action in enumerate(legal_actions):
        if action.to_dict().get("type") == "play_card":
            return i
    for i, action in enumerate(legal_actions):
        if action.to_dict().get("type") == "end_turn":
            return i
    return 0


def create_berserk_bot(
    profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> BerserkInference:
    """
    Фабрика для создания инстанса Берсерка с профилями сложности.

    Args:
        profiles: Словарь профилей {difficulty: {model_path, obs_dim, temperature_range}}

    Returns:
        Готовый к использованию BerserkInference
    """
    return BerserkInference(profiles=profiles)
