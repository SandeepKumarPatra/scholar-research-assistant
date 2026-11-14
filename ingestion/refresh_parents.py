#!/usr/bin/env python3
"""Metadata refresh script for Scholar papers.

Updates section_type in data/parsed/*.json and data/parents.db without touching Qdrant:
1. Creates a timestamped backup of data/parents.db.
2. Updates parsed JSONs: sections after References heading become 'appendix',
   and tables inherit section_type from their enclosing section.
3. Repopulates parents.db. If the parent_id set differs, automatically restores
   the backup and aborts with an error.
4. Prints a before/after comparison table of appendix and reference parent counts,
   including the percentage of paper tokens in references/appendices (with >40% warnings).
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from ingestion.chunk_structured import (
    DEFAULT_PARSED_DIR,
    DEFAULT_PARENTS_DB,
    chunk_all_parsed_papers,
    find_enclosing_section,
    save_parents_to_db,
)
from ingestion.parse_docling import classify_section_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scholar.refresh_parents")


def backup_database(db_path: Path) -> Optional[Path]:
    """Create a timestamped backup of the SQLite database."""
    if not db_path.exists():
        logger.warning(f"Database {db_path} does not exist. No backup created.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"parents_{timestamp}.db.bak"
    shutil.copy2(db_path, backup_path)
    logger.info(f"Backed up database to {backup_path}")
    return backup_path


def restore_database(backup_path: Path, target_path: Path) -> None:
    """Restore database from backup file."""
    if backup_path.exists():
        shutil.copy2(backup_path, target_path)
        logger.info(f"Restored database from {backup_path} to {target_path}")


def get_parents_summary(db_path: Path) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
    """Query parents.db for parent IDs and counts per paper."""
    if not db_path.exists():
        return set(), {}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT parent_id FROM parents")
    all_ids = set(row[0] for row in cursor.fetchall())

    cursor.execute(
        """
        SELECT paper_id, section_type, count(*), sum(token_count)
        FROM parents
        GROUP BY paper_id, section_type
        """
    )
    rows = cursor.fetchall()
    conn.close()

    summary: Dict[str, Dict[str, Any]] = {}
    for paper_id, sec_type, cnt, tok_sum in rows:
        summary.setdefault(
            paper_id,
            {"references": 0, "appendix": 0, "body": 0, "table": 0, "total_tokens": 0, "ref_app_tokens": 0},
        )
        tokens = tok_sum or 0
        summary[paper_id]["total_tokens"] += tokens
        if sec_type in ("references", "appendix"):
            summary[paper_id]["ref_app_tokens"] += tokens

        if sec_type == "references":
            summary[paper_id]["references"] += cnt
        elif sec_type == "appendix":
            summary[paper_id]["appendix"] += cnt
        elif sec_type == "body":
            summary[paper_id]["body"] += cnt
        else:
            summary[paper_id]["table"] += cnt

    return all_ids, summary


def update_parsed_json_files(parsed_dir: Path) -> Dict[str, bool]:
    """Update section_type in parsed JSON files. Returns dict of paper_id -> has_references."""
    json_files = sorted(parsed_dir.glob("*.json"))
    has_refs_map: Dict[str, bool] = {}

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {jf}: {e}")
            continue

        paper_id = data.get("paper_id", jf.stem)
        sections = data.get("sections", [])
        tables = data.get("tables", [])

        seen_references = False
        has_references_heading = False

        # 1. Update sections: strict exact match for references heading;
        # everything after references becomes appendix.
        for s in sections:
            heading = s.get("heading", "")
            ktype = classify_section_type(heading, past_references=seen_references)
            if ktype == "references":
                has_references_heading = True
                seen_references = True
                s["section_type"] = "references"
            elif seen_references:
                s["section_type"] = "appendix"
            else:
                s["section_type"] = ktype

        has_refs_map[paper_id] = has_references_heading

        # 2. Update tables: inherit section_type from enclosing section
        for t in tables:
            enclosing = find_enclosing_section(t, sections)
            t_sec_type = enclosing.get("section_type", "body") if enclosing else "body"
            t["section_type"] = t_sec_type

        # Save updated JSON back
        try:
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error writing {jf}: {e}")

    return has_refs_map


def refresh_parents_metadata(
    parsed_dir: Path = DEFAULT_PARSED_DIR,
    parents_db: Path = DEFAULT_PARENTS_DB,
) -> bool:
    """Execute metadata refresh on parsed JSONs and parents.db."""
    logger.info(f"Starting metadata refresh (parsed_dir={parsed_dir}, parents_db={parents_db})")

    # Step 1: Backup database
    backup_path = backup_database(parents_db)

    # Step 2: Record before state
    before_ids, before_summary = get_parents_summary(parents_db)

    # Step 3: Update parsed JSON files on disk
    has_refs_map = update_parsed_json_files(parsed_dir)

    # Step 4: Chunk parsed papers in-memory
    new_parents, _ = chunk_all_parsed_papers(
        parsed_dir=parsed_dir,
        parents_db=parents_db,
        save_to_db=False,
    )

    new_ids = set(p["parent_id"] for p in new_parents)

    # Step 5: Verify ID invariance
    if before_ids and new_ids != before_ids:
        missing_ids = before_ids - new_ids
        extra_ids = new_ids - before_ids
        logger.error(
            f"Parent ID set mismatch after refresh! "
            f"Expected {len(before_ids)} IDs, got {len(new_ids)}. "
            f"Missing: {len(missing_ids)}, Extra: {len(extra_ids)}."
        )
        if backup_path and backup_path.exists():
            restore_database(backup_path, parents_db)
            logger.error(f"Aborted and restored database from {backup_path}.")
        raise RuntimeError(
            f"Parent ID set changed during metadata refresh. Restored from {backup_path}."
        )

    # Step 6: Save updated parents into parents.db
    save_parents_to_db(new_parents, db_path=parents_db)
    logger.info(f"Successfully saved {len(new_parents)} updated parent records to {parents_db}")

    # Step 7: Record after state
    _, after_summary = get_parents_summary(parents_db)

    # Step 8: Print before / after comparison table
    print_refresh_report(before_summary, after_summary, has_refs_map)
    return True


def print_refresh_report(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    has_refs_map: Dict[str, bool],
) -> None:
    """Print formatted comparison table of metadata refresh."""
    all_papers = sorted(set(list(before.keys()) + list(after.keys())))

    print("\n" + "=" * 105)
    print("METADATA REFRESH: BEFORE / AFTER SUMMARY")
    print("=" * 105)
    header = (
        f"{'Paper ID':<18} | {'Refs (B -> A)':^15} | {'App (B -> A)':^15} | "
        f"{'Refs Found':^10} | {'Ref+App %':^12} | {'Notes'}"
    )
    print(header)
    print("-" * 105)

    total_refs_before = sum(b.get("references", 0) for b in before.values())
    total_refs_after = sum(a.get("references", 0) for a in after.values())
    total_app_before = sum(b.get("appendix", 0) for b in before.values())
    total_app_after = sum(a.get("appendix", 0) for a in after.values())
    total_papers_with_refs = sum(1 for v in has_refs_map.values() if v)

    for pid in all_papers:
        b_data = before.get(pid, {})
        a_data = after.get(pid, {})

        r_before = b_data.get("references", 0)
        r_after = a_data.get("references", 0)
        a_before = b_data.get("appendix", 0)
        a_after = a_data.get("appendix", 0)

        refs_str = f"{r_before:>2} -> {r_after:<2}"
        app_str = f"{a_before:>2} -> {a_after:<2}"
        has_ref = "Yes" if has_refs_map.get(pid, False) else "No"

        total_tok = a_data.get("total_tokens", 0)
        ref_app_tok = a_data.get("ref_app_tokens", 0)
        pct = (ref_app_tok / total_tok * 100.0) if total_tok > 0 else 0.0
        pct_str = f"{pct:5.1f}%"

        warning = "WARNING (>40%)" if pct > 40.0 else ""

        print(
            f"{pid:<18} | {refs_str:^15} | {app_str:^15} | "
            f"{has_ref:^10} | {pct_str:^12} | {warning}"
        )

    print("-" * 105)
    print(
        f"TOTALS: {len(all_papers)} papers | "
        f"References heading detected in {total_papers_with_refs}/{len(all_papers)} papers | "
        f"Appendix parents: {total_app_before} -> {total_app_after} | "
        f"Reference parents: {total_refs_before} -> {total_refs_after}"
    )
    print("=" * 105 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Refresh metadata in data/parsed/ and data/parents.db."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help=f"Directory with parsed paper JSONs (default: {DEFAULT_PARSED_DIR}).",
    )
    parser.add_argument(
        "--parents-db",
        type=Path,
        default=DEFAULT_PARENTS_DB,
        help=f"Path to SQLite database (default: {DEFAULT_PARENTS_DB}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for refresh_parents."""
    args = parse_args()
    try:
        refresh_parents_metadata(
            parsed_dir=args.parsed_dir,
            parents_db=args.parents_db,
        )
    except Exception as e:
        logger.error(f"Metadata refresh failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
