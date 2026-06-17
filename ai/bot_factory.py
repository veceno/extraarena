from __future__ import annotations

import logging
import random
from typing import Any

from infrastructure.config import (
    BOT_DIFFICULTY_PROFILES,
    BOT_DIFFICULTY_ALIASES,
    BOT_EXTRA_PASS_ROLL_PROBABILITIES,
    BOT_STRENGTH_TIERS,
    DECK_SIZE,
    get_bot_strength_tier,
    get_league_by_trophies_fn,
)


class BotGenerator:
    """
    BotFactory v2: переиспользование ботов из пула, ONNX, косметика, колоды.

    Подбирает боту имя, колоду, трофеи, сложность, косметику и сохраняет в БД.
    """

    DIFFICULTIES = tuple(BOT_DIFFICULTY_PROFILES.keys())

    def __init__(self, database: Any) -> None:
        self._db = database
        self._logger = logging.getLogger(__name__)
        self._fallback_metrics: dict[str, int] = {
            "recent_opponents_failed": 0,
            "find_reusable_failed": 0,
            "full_profile_failed": 0,
            "persist_failed": 0,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def get_or_create_bot(
        self,
        player_id: int,
        player_trophies: int,
        difficulty_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Try to reuse an existing bot; fallback to creation.
        Returns a unified bot payload.
        """
        if difficulty_override is not None:
            difficulty = self._normalize_difficulty(difficulty_override)
            return await self._generate_bot(
                player_id,
                player_trophies,
                difficulty_override=difficulty,
            )

        # 1. Collect recent bot opponents to exclude
        recent_bot_opponents: list[int] = []
        try:
            recent_bot_opponents = await self._db.get_recent_bot_opponents(player_id, limit=5)
        except Exception as exc:
            self._fallback_metrics["recent_opponents_failed"] += 1
            self._logger.warning(
                "get_recent_bot_opponents failed for player_id=%s: %s",
                player_id,
                exc,
                exc_info=True,
            )

        # 2. Try to find a reusable bot
        candidates = []
        try:
            candidates = await self._db.find_reusable_bots(
                player_trophies=player_trophies,
                exclude_user_ids=recent_bot_opponents,
                max_results=20,
            )
        except Exception as exc:
            self._fallback_metrics["find_reusable_failed"] += 1
            self._logger.warning(
                "find_reusable_bots failed for player_id=%s trophies=%s: %s",
                player_id,
                player_trophies,
                exc,
                exc_info=True,
            )

        for candidate in candidates:
            bot_id = candidate["user_id"]
            full = None
            try:
                full = await self._db.get_bot_full_profile(bot_id)
            except Exception as exc:
                self._fallback_metrics["full_profile_failed"] += 1
                self._logger.warning(
                    "get_bot_full_profile failed for bot_id=%s player_id=%s: %s",
                    bot_id,
                    player_id,
                    exc,
                    exc_info=True,
                )
            if not full or not full.get("deck_ids"):
                continue

            payload = await self._build_payload_from_profile(player_id, player_trophies, full, reused=True)
            if payload:
                self._logger.info(
                    "REUSED bot %s (trophies=%s) for player %s",
                    bot_id, payload.get("trophies"), player_id,
                )
                return payload

        # 3. Fallback: generate a new bot
        self._logger.info("No reusable bot found, generating new one for player %s", player_id)
        return await self._generate_bot(player_id, player_trophies)

    async def create_persistent_bot(
        self,
        player_id: int,
        player_trophies: int,
        player_avg_level: int,
    ) -> dict[str, Any]:
        """Legacy compatibility wrapper."""
        return await self.get_or_create_bot(player_id, player_trophies)

    # ------------------------------------------------------------------
    # Bot generation
    # ------------------------------------------------------------------

    async def _generate_bot(
        self,
        player_id: int,
        player_trophies: int,
        difficulty_override: str | None = None,
    ) -> dict[str, Any]:
        """Generate a brand new bot and persist everything."""
        # 1. Deck: try donor first
        deck_ids: list[int] = []
        bot_name = None
        bot_avatar_url = None
        difficulty = (
            self._normalize_difficulty(difficulty_override)
            if difficulty_override is not None
            else self._calc_difficulty(player_trophies)
        )
        deck_policy = self._difficulty_metadata(difficulty)["deck_policy"]

        donor_deck = await self._get_donor_deck(
            player_trophies,
            exclude_user_ids=[player_id],
            deck_policy=deck_policy,
        )
        if donor_deck:
            self._logger.info("Using donor deck: %s cards", len(donor_deck))
            deck_ids = donor_deck
        else:
            self._logger.warning("No donor deck, using random catalog deck")
            deck_ids = await self._build_bot_deck(player_trophies)
        deck_ids = await self._sanitize_deck(deck_ids)
        bot_name, bot_avatar_url = await self._get_random_donor_name(player_id)

        # 2. Trophies: player ± N, no hard cap, min 0
        n = max(25, min(round(player_trophies * 0.08), 500))
        trophy_delta = random.randint(-n, n)
        bot_trophies = max(0, player_trophies + trophy_delta)

        bot_league = get_league_by_trophies_fn(bot_trophies)

        # 4. Cosmetic picks
        cosmetics = await self._pick_bot_cosmetics()
        extra_pass = self._roll_extra_pass()

        # 5. Persist
        bot_id = 0
        display_name = bot_name or f"Бот {random.randint(1000, 9999)}"

        try:
            if hasattr(self._db, "create_generated_bot_profile"):
                persisted = await self._db.create_generated_bot_profile(
                    display_name=display_name,
                    trophies=bot_trophies,
                    level=1,
                    deck_ids=deck_ids,
                    avatar_url=bot_avatar_url,
                    extra_pass=extra_pass,
                    avatar_cos_id=cosmetics.get("avatar", {}).get("id"),
                    title_cos_id=cosmetics.get("title", {}).get("id"),
                    bg_cos_id=cosmetics.get("profile_background", {}).get("id"),
                )
                bot_id = int(persisted.get("bot_id", 0))
                if bot_id <= 0:
                    raise RuntimeError("create_generated_bot_profile returned invalid bot_id")
                self._logger.info("Created new bot %s trophies=%s difficulty=%s", bot_id, bot_trophies, difficulty)
            else:
                bot_id = await self._db.get_next_bot_id()
                self._logger.info("Creating new bot %s trophies=%s difficulty=%s", bot_id, bot_trophies, difficulty)
                await self._db.create_or_update_bot_profile(
                    bot_id=bot_id,
                    display_name=display_name,
                    trophies=bot_trophies,
                    level=1,
                    deck_ids=deck_ids,
                    avatar_url=bot_avatar_url,
                )
                # Persist extra_pass
                await self._db.execute(
                    "UPDATE users SET extra_pass = $2 WHERE user_id = $1",
                    bot_id, extra_pass,
                )
                # Persist cosmetics
                await self._db.grant_and_equip_bot_cosmetics(
                    bot_id=bot_id,
                    avatar_cos_id=cosmetics.get("avatar", {}).get("id"),
                    title_cos_id=cosmetics.get("title", {}).get("id"),
                    bg_cos_id=cosmetics.get("profile_background", {}).get("id"),
                )
        except Exception as exc:
            self._fallback_metrics["persist_failed"] += 1
            self._logger.error("Failed to persist bot %s: %s", bot_id, exc, exc_info=True)
            return self._build_fallback_payload(bot_id, deck_ids, bot_name, bot_avatar_url,
                                                bot_trophies, difficulty, cosmetics, extra_pass)

        return self._build_payload(
            bot_id=bot_id,
            deck_ids=deck_ids,
            bot_name=bot_name,
            bot_avatar_url=bot_avatar_url,
            bot_trophies=bot_trophies,
            difficulty=difficulty,
            bot_league=bot_league,
            cosmetics=cosmetics,
            extra_pass=extra_pass,
            reused=False,
        )

    # ------------------------------------------------------------------
    # Donor deck helpers
    # ------------------------------------------------------------------

    async def _get_donor_deck(
        self,
        player_trophies: int,
        exclude_user_ids: list[int] | None = None,
        deck_policy: str = "donor",
    ) -> list[int] | None:
        if deck_policy == "starter_random":
            return None

        trophy_offsets = {
            "weak_donor": -200,
            "donor": 0,
            "similar_donor": 0,
            "decent_donor": 75,
            "donor_basic_synergy": 125,
            "strong_donor": 250,
            "curated_donor": 350,
            "strong_meta": 500,
            "meta": 650,
            "meta_boss": 900,
        }
        lookup_trophies = max(0, int(player_trophies) + trophy_offsets.get(deck_policy, 0))
        try:
            deck = await self._db.get_bot_deck_from_donor(lookup_trophies, exclude_user_ids)
            if deck and await self._is_valid_donor_deck(deck):
                return deck
        except Exception as exc:
            self._logger.warning(
                "Donor deck lookup failed policy=%s trophies=%s lookup=%s: %s",
                deck_policy,
                player_trophies,
                lookup_trophies,
                exc,
                exc_info=True,
            )
        return None

    @staticmethod
    def _is_valid_deck(deck_ids: list[int]) -> bool:
        return len(deck_ids or []) == DECK_SIZE

    async def _is_valid_donor_deck(self, deck_ids: list[int]) -> bool:
        """Validate basic deck shape; use card catalog to require a hero when available."""
        if not self._is_valid_deck(deck_ids):
            return False
        try:
            cards_catalog = await self._db.get_cards_list()
            type_by_id: dict[int, str] = {}
            for item in cards_catalog:
                c_id = getattr(item, "id", None) or item.get("id")
                c_type = getattr(item, "card_type", None) or item.get("card_type", "unit")
                if c_id is not None:
                    type_by_id[int(c_id)] = str(c_type).lower()
            known_types = [type_by_id.get(int(card_id), "unit") for card_id in deck_ids]
            hero_count = sum(1 for card_type in known_types if card_type == "hero")
            return hero_count == 1
        except Exception as exc:
            self._logger.warning("donor deck validation failed: %s", exc, exc_info=True)
            return False

    async def _get_random_donor_name(
        self, exclude_user_id: int,
    ) -> tuple[str, str | None]:
        try:
            users = await self._db.get_random_users_with_avatars(10, exclude_user_id)
            if users:
                pick = random.choice(users)
                return pick.get("display_name", f"Бот {random.randint(1000, 9999)}"), pick.get("img")
        except Exception as exc:
            self._logger.warning("random donor name lookup failed: %s", exc, exc_info=True)
        name = f"Бот {random.randint(1000, 9999)}"
        return name, None

    # ------------------------------------------------------------------
    # Deck fallback
    # ------------------------------------------------------------------

    async def _get_disabled_card_ids(self) -> set[int]:
        try:
            if hasattr(self._db, "get_disabled_card_ids"):
                return {int(card_id) for card_id in await self._db.get_disabled_card_ids()}
        except Exception as exc:
            self._logger.warning("disabled card lookup failed: %s", exc)
        return set()

    async def _card_catalog_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            cards_catalog = await self._db.get_cards_list()
        except Exception as exc:
            self._logger.warning("card catalog lookup failed: %s", exc)
            return rows

        for item in cards_catalog or []:
            c_id = getattr(item, "id", None) if not isinstance(item, dict) else item.get("id")
            c_type = getattr(item, "card_type", None) if not isinstance(item, dict) else item.get("card_type", "unit")
            if c_id is not None:
                rows.append({"id": int(c_id), "type": str(c_type or "unit").lower()})
        return rows

    async def _sanitize_deck(self, deck_ids: list[int]) -> list[int]:
        disabled = await self._get_disabled_card_ids()
        catalog = await self._card_catalog_rows()
        if not catalog:
            raise ValueError("Cannot build valid bot deck: card catalog is unavailable")
        target_size = DECK_SIZE
        target_non_heroes = max(0, target_size - 1)

        type_by_id = {row["id"]: row["type"] for row in catalog}
        allowed_heroes: list[int] = []
        allowed_non_heroes: list[int] = []
        for row in catalog:
            if row["id"] in disabled:
                continue
            if row["type"] == "hero":
                allowed_heroes.append(row["id"])
            else:
                allowed_non_heroes.append(row["id"])

        if not allowed_heroes or len(allowed_non_heroes) < target_non_heroes:
            raise ValueError("Cannot build valid bot deck: insufficient valid bot deck pool")

        used: set[int] = set()
        hero_id: int | None = None
        non_heroes: list[int] = []
        for raw_card_id in deck_ids or []:
            card_id = int(raw_card_id)
            if card_id in disabled or card_id not in type_by_id or card_id in used:
                continue
            if type_by_id[card_id] == "hero":
                if hero_id is None:
                    hero_id = card_id
                    used.add(card_id)
                continue
            if len(non_heroes) < target_non_heroes:
                non_heroes.append(card_id)
                used.add(card_id)

        if hero_id is None:
            hero_candidates = [cid for cid in allowed_heroes if cid not in used] or allowed_heroes
            hero_id = random.choice(hero_candidates)
            used.add(hero_id)

        while len(non_heroes) < target_non_heroes:
            candidates = [cid for cid in allowed_non_heroes if cid not in used]
            if not candidates:
                raise ValueError("Cannot build valid bot deck: not enough unique non-hero cards")
            replacement = random.choice(candidates)
            non_heroes.append(replacement)
            used.add(replacement)

        sanitized = [hero_id] + non_heroes[:target_non_heroes]
        random.shuffle(sanitized)

        hero_count = sum(1 for card_id in sanitized if type_by_id.get(card_id) == "hero")
        if len(sanitized) != target_size or hero_count != 1 or any(card_id in disabled for card_id in sanitized):
            raise ValueError("Cannot build valid bot deck")
        return sanitized

    async def _build_bot_deck(self, player_trophies: int) -> list[int]:
        deck_ids: list[int] = []
        try:
            disabled = await self._get_disabled_card_ids()
            all_cards = [card for card in await self._card_catalog_rows() if card["id"] not in disabled]

            heroes = [c["id"] for c in all_cards if c["type"] == "hero"]
            units = [c["id"] for c in all_cards if c["type"] != "hero"]

            if heroes:
                deck_ids.append(random.choice(heroes))
            random.shuffle(units)
            needed = DECK_SIZE - 1
            deck_ids.extend(units[:min(len(units), needed)])
            random.shuffle(deck_ids)
        except Exception as exc:
            self._logger.error("_build_bot_deck failed: %s", exc, exc_info=True)
        return deck_ids

    # ------------------------------------------------------------------
    # Difficulty
    # ------------------------------------------------------------------

    @classmethod
    def _calc_difficulty(cls, player_trophies: int) -> str:
        """
        Map player trophies to an internal strength tier.
        Card levels are now built per-card by _build_bot_card_levels.
        """
        return str(get_bot_strength_tier(player_trophies)["key"])

    @classmethod
    def _calc_public_difficulty(cls, player_trophies: int) -> str:
        return str(get_bot_strength_tier(player_trophies)["difficulty_label"])

    @classmethod
    def _shift_difficulty_by_streak(
        cls,
        difficulty: str,
        direction: str | None,
        n: int,
    ) -> str:
        """Shift a bot strength tier by streak pressure without changing trophy-road mapping."""
        resolved = cls._normalize_difficulty(difficulty)
        if direction not in {"up", "down"}:
            return resolved

        steps = min(5, max(0, int(n or 0)))
        if steps <= 0:
            return resolved

        tier_keys = [str(tier["key"]) for tier in BOT_STRENGTH_TIERS]
        current_idx = tier_keys.index(resolved)
        delta = steps if direction == "up" else -steps
        shifted_idx = max(0, min(len(tier_keys) - 1, current_idx + delta))
        return tier_keys[shifted_idx]

    @staticmethod
    def _tier_by_key(difficulty: str) -> dict[str, Any] | None:
        difficulty = BOT_DIFFICULTY_ALIASES.get(str(difficulty), str(difficulty))
        for tier in BOT_STRENGTH_TIERS:
            if tier["key"] == difficulty:
                return tier
        return None

    @classmethod
    def _normalize_difficulty(cls, difficulty: str) -> str:
        tier = cls._tier_by_key(difficulty)
        if tier is None:
            raise KeyError(f"Unknown bot difficulty/tier: {difficulty}")
        return str(tier["key"])

    @classmethod
    def _match_difficulty(cls, player_trophies: int, difficulty: str | None = None) -> str:
        if difficulty:
            try:
                return cls._normalize_difficulty(str(difficulty))
            except KeyError:
                pass
        return cls._calc_difficulty(player_trophies)

    @classmethod
    def _difficulty_metadata(cls, difficulty: str) -> dict[str, Any]:
        tier = cls._tier_by_key(difficulty)
        if tier is None:
            raise KeyError(f"Unknown bot difficulty/tier: {difficulty}")
        return {
            "difficulty_label": tier["difficulty_label"],
            "strength_tier": tier["key"],
            "brain_profile": tier["brain_profile"],
            "selection": tier["selection"],
            "temperature": tier["temperature"],
            "card_level_policy": dict(tier["level_policy"]),
            "deck_policy": tier["deck_policy"],
        }

    @classmethod
    def normalize_bot_info(
        cls,
        bot_info: dict[str, Any],
        *,
        player_trophies: int,
        user_max_level: int,
        difficulty: str | None = None,
    ) -> dict[str, Any]:
        """Fill match-scoped bot difficulty metadata and per-card levels."""
        normalized = dict(bot_info or {})
        resolved_difficulty = cls._match_difficulty(
            player_trophies,
            difficulty or normalized.get("difficulty"),
        )
        deck_count = len(normalized.get("deck_ids") or [])
        card_levels = normalized.get("card_levels")
        incoming_difficulty = normalized.get("difficulty")
        difficulty_changed = False
        if difficulty is not None and incoming_difficulty:
            try:
                difficulty_changed = cls._normalize_difficulty(str(incoming_difficulty)) != resolved_difficulty
            except KeyError:
                difficulty_changed = True

        normalized["difficulty"] = resolved_difficulty
        normalized.update(cls._difficulty_metadata(resolved_difficulty))
        if difficulty_changed or not isinstance(card_levels, list) or len(card_levels) != deck_count:
            normalized["card_levels"] = cls._build_bot_card_levels(
                resolved_difficulty,
                int(user_max_level or 1),
                deck_count,
            )
        return normalized

    @staticmethod
    def _build_bot_card_levels(
        difficulty: str, player_max_level: int, deck_card_count: int
    ) -> list[int]:
        """Build slot-based bot card levels from the trophy-road strength policy."""
        if deck_card_count <= 0:
            return []

        meta = BotGenerator._difficulty_metadata(difficulty)
        policy = meta["card_level_policy"]
        lo = int(policy.get("delta_min", 0))
        hi = int(policy.get("delta_max", lo))
        cap = int(policy.get("cap", 10))
        boost_fraction = float(policy.get("boost_fraction", 0.0))

        levels = [
            max(1, min(cap, player_max_level + random.randint(lo, hi)))
            for _ in range(deck_card_count)
        ]

        boost_count = max(0, min(deck_card_count, round(deck_card_count * boost_fraction)))
        if boost_count:
            indices = list(range(deck_card_count))
            random.shuffle(indices)
            for idx in indices[:boost_count]:
                levels[idx] = max(1, min(cap, levels[idx] + 1))

        return levels

    # ------------------------------------------------------------------
    # Cosmetics
    # ------------------------------------------------------------------

    CLASS_WEIGHTS = {
        "starter": 0.55,
        "rare": 0.25,
        "epic": 0.13,
        "mythic": 0.05,
        "mythical": 0.05,
        "limited": 0.02,
    }

    async def _pick_bot_cosmetics(self) -> dict[str, dict[str, Any]]:
        """Pick avatar, title, background with rarity weights."""
        result: dict[str, dict[str, Any]] = {}
        try:
            catalog = await self._db.get_cosmetic_catalog_by_class()
            if not catalog:
                return result

            for item_type in ("avatar", "profile_background", "title"):
                chosen = self._weighted_pick_by_class(catalog, item_type)
                if chosen:
                    result[item_type] = chosen
        except Exception as exc:
            self._logger.warning("Cosmetic catalog fetch failed: %s", exc)
        return result

    def _weighted_pick_by_class(
        self, catalog: dict[str, list[dict]], item_type: str,
    ) -> dict[str, Any] | None:
        """Weighted random pick from catalog by class, filtered by item_type."""
        candidates: list[tuple[dict, float]] = []
        for cls_name, items in catalog.items():
            weight = self.CLASS_WEIGHTS.get(cls_name, 0.01)
            matching = [i for i in items if i.get("item_type") == item_type]
            for item in matching:
                candidates.append((item, weight))

        if not candidates:
            return None

        total = sum(w for _, w in candidates)
        r = random.random() * total
        cumulative = 0.0
        for item, w in candidates:
            cumulative += w
            if r <= cumulative:
                return item
        return candidates[-1][0] if candidates else None

    @staticmethod
    def _roll_extra_pass() -> str:
        probabilities = dict(BOT_EXTRA_PASS_ROLL_PROBABILITIES)
        total = sum(max(0.0, float(v)) for v in probabilities.values())
        if total <= 0:
            return "inactive"

        r = random.random() * total
        cumulative = 0.0
        for status in ("ultra", "active", "inactive"):
            cumulative += max(0.0, float(probabilities.get(status, 0.0)))
            if r <= cumulative:
                return status
        return "inactive"

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    async def _build_payload_from_profile(
        self,
        player_id: int,
        player_trophies: int,
        profile: dict[str, Any],
        reused: bool,
    ) -> dict[str, Any] | None:
        """Build a unified bot payload from a reused DB bot profile."""
        bot_id = profile["user_id"]
        bot_trophies = profile.get("trophies", player_trophies)
        original_deck_ids = [int(card_id) for card_id in profile.get("deck_ids", [])]
        if not self._is_valid_deck(original_deck_ids):
            return None
        try:
            deck_ids = await self._sanitize_deck(original_deck_ids)
        except Exception as exc:
            self._logger.warning("Reused bot %s deck sanitization failed: %s", bot_id, exc, exc_info=True)
            return None
        if not self._is_valid_deck(deck_ids):
            return None
        if sorted(deck_ids) != sorted(original_deck_ids) and hasattr(self._db, "update_bot_deck_preset"):
            try:
                await self._db.update_bot_deck_preset(bot_id, deck_ids)
            except Exception as exc:
                self._logger.warning("Failed to persist repaired deck for reused bot %s: %s", bot_id, exc)

        bot_name = profile.get("display_name") or f"Бот {bot_id}"
        bot_avatar_url = profile.get("img")
        extra_pass = profile.get("extra_pass", "inactive")
        difficulty = self._calc_difficulty(player_trophies)
        bot_league = profile.get("league", get_league_by_trophies_fn(bot_trophies))

        # Check and possibly patch missing cosmetics
        equipped = profile.get("equipped_cosmetics", {})
        cosmetics: dict[str, dict[str, Any]] = {}
        needs_cosmetics_patch = (
            "avatar" not in equipped or "title" not in equipped or "profile_background" not in equipped
        )
        if needs_cosmetics_patch:
            try:
                missing = await self._pick_bot_cosmetics()
                await self._db.grant_and_equip_bot_cosmetics(
                    bot_id=bot_id,
                    avatar_cos_id=missing.get("avatar", {}).get("id"),
                    title_cos_id=missing.get("title", {}).get("id"),
                    bg_cos_id=missing.get("profile_background", {}).get("id"),
                )
                cosmetics = missing
            except Exception as exc:
                self._logger.warning("Failed to patch cosmetics for reused bot %s: %s", bot_id, exc)
                cosmetics = {}
        else:
            cosmetics = equipped

        return self._build_payload(
            bot_id=bot_id, deck_ids=deck_ids, bot_name=bot_name,
            bot_avatar_url=bot_avatar_url, bot_trophies=bot_trophies,
            difficulty=difficulty, bot_league=bot_league,
            cosmetics=cosmetics, extra_pass=extra_pass, reused=True,
        )

    @staticmethod
    def _require_full_bot_deck(deck_ids: list[int]) -> None:
        if len(deck_ids or []) != DECK_SIZE:
            raise ValueError(f"Bot payload requires a full bot deck of {DECK_SIZE} cards")

    @staticmethod
    def _build_payload(
        bot_id: int,
        deck_ids: list[int],
        bot_name: str | None,
        bot_avatar_url: str | None,
        bot_trophies: int,
        difficulty: str,
        bot_league: int,
        cosmetics: dict[str, dict[str, Any]],
        extra_pass: str,
        reused: bool,
    ) -> dict[str, Any]:
        BotGenerator._require_full_bot_deck(deck_ids)
        return {
            "user_id": bot_id,
            "deck_ids": deck_ids,
            "name": bot_name or f"Бот {bot_id}",
            "avatar_url": bot_avatar_url,
            "difficulty": difficulty,
            **BotGenerator._difficulty_metadata(difficulty),
            "trophies": bot_trophies,
            "league": bot_league,
            "extra_pass": extra_pass,
            "cosmetics": {
                "avatar": cosmetics.get("avatar"),
                "title": cosmetics.get("title"),
                "profile_background": cosmetics.get("profile_background"),
            },
            "reused": reused,
            "persisted": True,
        }

    @staticmethod
    def _build_fallback_payload(
        bot_id: int,
        deck_ids: list[int],
        bot_name: str | None,
        bot_avatar_url: str | None,
        bot_trophies: int,
        difficulty: str,
        cosmetics: dict[str, dict[str, Any]],
        extra_pass: str,
    ) -> dict[str, Any]:
        BotGenerator._require_full_bot_deck(deck_ids)
        temp_bot_id = -abs(int(bot_id) or random.randint(1, 1_000_000))
        return {
            "user_id": temp_bot_id,
            "source_bot_id": bot_id,
            "deck_ids": deck_ids,
            "name": bot_name or f"Бот {bot_id}",
            "avatar_url": bot_avatar_url,
            "difficulty": difficulty,
            **BotGenerator._difficulty_metadata(difficulty),
            "trophies": bot_trophies,
            "league": get_league_by_trophies_fn(bot_trophies),
            "extra_pass": extra_pass,
            "cosmetics": {
                "avatar": cosmetics.get("avatar"),
                "title": cosmetics.get("title"),
                "profile_background": cosmetics.get("profile_background"),
            },
            "reused": False,
            "persisted": False,
        }
