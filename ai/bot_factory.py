from __future__ import annotations

import logging
import random
from typing import Any

from infrastructure.database import Database


class BotGenerator:
    """
    Генератор постоянных (persisted) ботов.
    Подбирает боту имя, колоду и параметры сложности на основе показателей игрока,
    а затем сохраняет бота в БД, чтобы его можно было использовать в матчмейкинге и истории боёв.
    """

    def __init__(self, database: Database) -> None:
        # Храним ссылку на Database, чтобы использовать готовые методы и единый пул подключений.
        self._db = database
        # Логгер для детальной диагностики сбоев генерации бота.
        self._logger = logging.getLogger(__name__)

    async def _build_bot_deck(self, player_trophies: int) -> list[int]:
        """
        Собирает колоду бота из 9 карт: 1 Герой + 8 Юнитов/Заклинаний.
        
        Логика:
        1) Получаем полный каталог карт.
        2) Разделяем на героев (card_type='hero') и остальных.
        3) Выбираем 1 случайного героя.
        4) Выбираем 8 случайных карт из остальных.
        """
        deck_ids: list[int] = []

        self._logger.info("DEBUG: _build_bot_deck start player_trophies=%s", player_trophies)

        try:
            # 1. Получаем каталог
            cards_catalog = await self._db.get_cards_list()
            
            # Приводим к списку объектов или dict (в зависимости от реализации БД)
            all_cards = []
            for item in cards_catalog:
                # Нормализация для удобства фильтрации
                c_id = getattr(item, "id", None) or item.get("id")
                c_type = getattr(item, "card_type", None) or item.get("card_type", "unit")
                
                if c_id is not None:
                    all_cards.append({"id": int(c_id), "type": str(c_type).lower()})

            # 2. Разделяем на героев и юнитов
            heroes = [c["id"] for c in all_cards if c["type"] == "hero"]
            units = [c["id"] for c in all_cards if c["type"] != "hero"]
            
            self._logger.info("DEBUG: Catalog split: heroes=%d, units=%d", len(heroes), len(units))

            # 3. Выбираем 1 героя
            if heroes:
                hero_id = random.choice(heroes)
                deck_ids.append(hero_id)
            else:
                self._logger.warning("DEBUG: Heroes list is empty! Bot will play without hero card.")

            # 4. Выбираем 8 юнитов (8 боевых карт + 1 герой = 9 всего)
            random.shuffle(units)
            
            needed_units = 8
            # Если юнитов меньше чем надо, берем сколько есть
            count_to_take = min(len(units), needed_units)
            deck_ids.extend(units[:count_to_take])

            # Лог результата
            self._logger.info(
                "DEBUG: _build_bot_deck assembled: hero_present=%s, total_cards=%s", 
                bool(heroes), 
                len(deck_ids)
            )

        except Exception as exc:  # noqa: BLE001
            self._logger.error("Не удалось собрать колоду для бота: %s", exc, exc_info=True)
            # Фолбэк: вернем пустой список, движок сам попытается что-то сделать или упадет
            return []

        # На всякий случай перемешиваем, хотя герой обычно фильтруется движком отдельно
        # Но лучше, чтобы порядок в списке (кроме героя) был случайным
        # Героя лучше оставить первым или не важно, движок ищет по типу.
        # random.shuffle(deck_ids) -> перемешивать весь список опасно, если движок ждет героя первым?
        # В battle_engine.py: _resolve_hero_data ищет по card_type, так что порядок не важен.
        random.shuffle(deck_ids)
        
        return deck_ids

    async def create_persistent_bot(
        self,
        player_id: int,
        player_trophies: int,
        player_avg_level: int,
    ) -> dict[str, Any]:
        """
        Создает бота-донора на основе профиля реального игрока.

        Логика:
        1) Ищем случайного донора с готовой колодой через get_random_donor_profile.
        2) Копируем имя, аватар и колоду донора.
        3) Рассчитываем трофеи: player_trophies ± 25 (не выше 300 для classic режима).
        4) Определяем сложность: 0-50 (noob), 51-200 (easy), 201-300 (medium), >300 (hard).
        5) Сохраняем бота в users с ID 810416XXXX и is_bot=True.
        """
        self._logger.info(
            "DEBUG: create_persistent_bot start player_id=%s trophies=%s avg_level=%s",
            player_id,
            player_trophies,
            player_avg_level,
        )

        # 1. Получаем профиль донора (имя, аватар, колода)
        donor_profile = await self._db.get_random_donor_profile(player_id)
        
        if donor_profile and donor_profile.get("deck_ids"):
            # Используем колоду донора
            deck_ids = donor_profile["deck_ids"]
            bot_name = donor_profile["display_name"]
            bot_avatar_url = donor_profile.get("img")
            donor_avg_level = donor_profile.get("avg_level", 1)
            self._logger.info(
                "DEBUG: Using donor profile: name=%s, deck_size=%s, avg_level=%s",
                bot_name, len(deck_ids), donor_avg_level
            )
        else:
            # Fallback: генерируем случайную колоду
            self._logger.warning("DEBUG: No donor found, generating random deck")
            deck_ids = await self._build_bot_deck(player_trophies)
            candidate_users = await self._db.get_random_users_with_avatars(20, player_id)
            
            if candidate_users:
                donor_user = random.choice(candidate_users)
                bot_name = donor_user.get("display_name") or f"Бот {random.randint(1000, 9999)}"
                bot_avatar_url = donor_user.get("img")
            else:
                bot_name = f"Бот {random.randint(1000, 9999)}"
                bot_avatar_url = None
            donor_avg_level = player_avg_level

        # 2. Рассчитываем трофеи: player_trophies ± 25, не выше лимита режима (300)
        from infrastructure.config import MM_TROPHY_LIMIT_CLASSIC
        trophy_delta = random.randint(-25, 25)
        bot_trophies = max(0, min(player_trophies + trophy_delta, MM_TROPHY_LIMIT_CLASSIC))

        # 3. Определяем сложность на основе трофеев
        # КРИТИЧНО: Для игроков с 0-300 трофеев используем только lite/easy
        if player_trophies < MM_TROPHY_LIMIT_CLASSIC:
            # Новички (0-300 трофеев) - только легкие боты
            if player_trophies == 0:
                # Для 0 трофеев - самая легкая сложность
                difficulty = "lite"
                level_adjustment = -1
            elif player_trophies < 100:
                # До 100 трофеев - lite
                difficulty = "lite"
                level_adjustment = -1
            else:
                # 100-299 трофеев - easy
                difficulty = "easy"
                level_adjustment = 0
        else:
            # Опытные игроки (300+ трофеев) - полный спектр сложности
            if bot_trophies <= 50:
                difficulty = "noob"
                level_adjustment = -1
            elif bot_trophies <= 200:
                difficulty = "easy"
                level_adjustment = 0
            elif bot_trophies <= 300:
                difficulty = "medium"
                level_adjustment = 0
            else:
                difficulty = "hard"
                level_adjustment = 1

        # Уровень бота базируется на среднем уровне донора с поправкой на сложность
        bot_level = max(1, donor_avg_level + level_adjustment)
        
        self._logger.info(
            "DEBUG: bot params bot_trophies=%s difficulty=%s bot_level=%s name=%s",
            bot_trophies, difficulty, bot_level, bot_name
        )

        # 4. Создаем бота в БД с ID в диапазоне 810416XXXX
        bot_id = await self._db.get_next_bot_id()
        try:
            self._logger.info(
                "DEBUG: create_or_update_bot_profile call bot_id=%s deck_len=%s avatar_url=%s",
                bot_id, len(deck_ids), bot_avatar_url,
            )
            await self._db.create_or_update_bot_profile(
                bot_id=bot_id,
                display_name=bot_name,
                trophies=bot_trophies,
                level=bot_level,
                deck_ids=deck_ids,
                avatar_url=bot_avatar_url,
            )
            self._logger.info("DEBUG: create_or_update_bot_profile OK bot_id=%s", bot_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("create_or_update_bot_profile упал для bot_id=%s: %s", bot_id, exc, exc_info=True)
            return {
                "user_id": bot_id,
                "deck_ids": deck_ids or [],
                "level": bot_level,
                "name": bot_name,
                "avatar_url": bot_avatar_url,
                "difficulty": difficulty,
            }

        # Возвращаем структуру бота
        payload = {
            "user_id": bot_id,
            "deck_ids": deck_ids,
            "level": bot_level,
            "name": bot_name,
            "avatar_url": bot_avatar_url,
            "difficulty": difficulty,
        }
        self._logger.info("DEBUG: create_persistent_bot result=%s", payload)
        return payload






