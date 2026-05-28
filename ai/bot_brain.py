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
                for key in ("include_preview_features", "verify_mask", "placement_mode"):
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

        include_preview = bool(profile.get("include_preview_features", True))
        return input_names, output_names, action_feature_dim, max_candidate_actions, include_preview

    def __init__(
        self,
        profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        action_dim: int = 200,
    ):
        """
        Args:
            profiles: Словарь профилей вида {difficulty: {model_path, obs_dim, temperature_range}}
            action_dim: Размерность вектора действий (200 макс. действий)
        """
        self.action_dim = action_dim
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        if profiles is None:
            profiles = {}
        
        # Загружаем все профили. Несколько difficulty/tier keys могут ссылаться
        # на один ONNX-файл с разной temperature/selection, поэтому сессии
        # переиспользуются по пути модели.
        session_cache: Dict[str, ort.InferenceSession] = {}
        load_errors: list[tuple[str, Exception]] = []
        for difficulty, profile in profiles.items():
            model_path = self._resolve_model_path(profile["model_path"])
            
            if not model_path.exists():
                exc = FileNotFoundError(f"{difficulty}: model not found: {model_path}")
                logger.error("[BerserkInference] %s", exc)
                load_errors.append((difficulty, exc))
                continue
            
            try:
                sidecar = self._load_sidecar(model_path)
                merged_profile = dict(sidecar)
                merged_profile.update(profile)
                profile_format = merged_profile.get("format", None)
                if profile_format != _TRAIN_V2_FORMAT:
                    logger.warning(
                        "[BerserkInference] %s skipped: legacy/non-TrainV2 profile format=%s",
                        difficulty,
                        profile_format,
                    )
                    continue

                temp_range = self._validate_temperature_range(difficulty, merged_profile)
                merged_profile["temperature_range"] = temp_range

                cache_key = str(model_path.resolve())
                session = session_cache.get(cache_key)
                if session is None:
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=["CPUExecutionProvider"],
                    )
                    session_cache[cache_key] = session

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
                    "selection": merged_profile.get("selection", "argmax"),
                    "placement_mode": merged_profile.get("placement_mode", "append_only"),
                    "verify_mask": merged_profile.get("verify_mask", True),
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
        include_preview = bool(profile.get("include_preview_features", True))
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

        if selection_mode == "argmax":
            action_id = int(np.argmax(mlogits))
        else:
            temperature = random.uniform(temp_min, temp_max)
            legal_mask = mask.astype(bool)
            if not np.any(legal_mask):
                logger.warning("[BerserkInference] TrainV2 mask has no legal actions, fallback")
                return _legal_fallback(legal_actions)
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

def _legal_fallback(legal_actions: list) -> int:
    for i, a in enumerate(legal_actions):
        if a.to_dict().get("type") == "end_turn":
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
