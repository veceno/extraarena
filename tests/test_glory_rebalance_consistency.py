"""Sanity-тесты для перебалансировки Glory Path (трофейной дороги).

Эти тесты проверяют, что seed-список ``glory`` в ``_seed_reward_tracks``
и канонический набор в ``_migrate_reward_tracks_glory_rebalance`` согласованы
для позиций, по которым был баланс. Это страховка от того, что кто-то
поправит одну таблицу и забудет про вторую, и в БД снова появятся старые
награды.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PY = REPO_ROOT / "infrastructure" / "database.py"


def _load_module_ast() -> ast.Module:
    return ast.parse(DATABASE_PY.read_text(encoding="utf-8"))


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {DATABASE_PY}")


def _glory_rows_from_seed(func: ast.FunctionDef) -> list[tuple]:
    """Извлечь список кортежей из ``rows = [...]`` в ``_seed_reward_tracks``."""
    for stmt in func.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == "rows":
                if not isinstance(stmt.value, ast.List):
                    raise AssertionError("'rows' should be a list literal")
                rows = []
                for elt in stmt.value.elts:
                    if not isinstance(elt, ast.Tuple):
                        continue
                    rows.append(tuple(_literal(elt.elts[i]) for i in range(len(elt.elts))))
                return rows
    raise AssertionError("'rows = [...]' assignment not found in _seed_reward_tracks")


def _literal(node):
    """Раскрыть AST-литерал в python-значение (только то, что реально встречается в seed)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id == "None":
        return None
    if isinstance(node, ast.Name) and node.id in ("True", "False"):
        return node.id == "True"
    raise AssertionError(f"unsupported literal node: {ast.dump(node)}")


def _canonical_rows_from_migration(func: ast.FunctionDef) -> list[tuple]:
    """Извлечь ``canonical`` из тела миграции."""
    for stmt in func.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "canonical":
            if not isinstance(stmt.value, ast.List):
                raise AssertionError("'canonical' should be a list literal")
            rows = []
            for elt in stmt.value.elts:
                if not isinstance(elt, ast.Tuple):
                    continue
                rows.append(tuple(_literal(elt.elts[i]) for i in range(len(elt.elts))))
            return rows
    raise AssertionError("'canonical: [...]' assignment not found in migration")


def test_glory_seed_and_migration_are_consistent():
    """Набор наград для позиций 1000/1500/3000/7000/10000 в seed и в
    каноническом списке миграции должен совпадать — иначе миграция перепишет
    seed (или наоборот) и в БД появится дрейф.
    """
    module = _load_module_ast()
    seed_func = _find_function(module, "_seed_reward_tracks")
    migrate_func = _find_function(module, "_migrate_reward_tracks_glory_rebalance")

    REBALANCED_POSITIONS = {1000, 1500, 3000, 7000, 10000}

    seed_rows = [r for r in _glory_rows_from_seed(seed_func) if r[0] == "glory" and r[1] in REBALANCED_POSITIONS]
    canonical_rows = [r for r in _canonical_rows_from_migration(migrate_func) if r[1] in REBALANCED_POSITIONS]

    seed_set = set(seed_rows)
    canonical_set = set(canonical_rows)
    assert seed_set == canonical_set, (
        "Glory rebalance seed и migration разъехались:\n"
        f"  только в seed: {seed_set - canonical_set}\n"
        f"  только в migration: {canonical_set - seed_set}"
    )


def test_glory_migration_canonical_has_no_legacy_case_rewards():
    """Миграция не должна оставлять ``case``-награды на ребаланс-позициях.

    Защита от регрессии: если кто-то добавит строку в ``canonical`` в обход
    плана перебалансировки (например, случайно вернёт ``case`` на 10000),
    этот тест укажет на проблему.
    """
    module = _load_module_ast()
    migrate_func = _find_function(module, "_migrate_reward_tracks_glory_rebalance")
    canonical_rows = _canonical_rows_from_migration(migrate_func)

    for row in canonical_rows:
        track_type, position, reward_type, _amount, _meta, _ep = row
        assert track_type == "glory", f"canonical содержит не-glory трек: {row}"
        assert reward_type in {"coins", "gems", "keys"}, (
            f"на позиции {position} миграция содержит неразрешённый reward_type={reward_type!r}; "
            f"для позиций 1000/1500/3000/7000/10000 по плану допустимы только coins/gems/keys"
        )


def test_glory_migration_called_from_ensure_table():
    """``_migrate_reward_tracks_glory_rebalance`` должна вызываться из
    ``_ensure_reward_tracks_table``, иначе миграция не отработает при
    рестарте web-сервиса.
    """
    module = _load_module_ast()
    ensure_func = _find_function(module, "_ensure_reward_tracks_table")
    body_src = ast.dump(ensure_func)
    assert "_migrate_reward_tracks_glory_rebalance" in body_src, (
        "_migrate_reward_tracks_glory_rebalance не вызывается из _ensure_reward_tracks_table"
    )
