#!/usr/bin/env python3
"""
Password Migration Script
Migrates plaintext passwords to bcrypt hashed passwords

Usage:
    python scripts/migrate_passwords.py [--dry-run]

Options:
    --dry-run    Show what would be changed without making changes
"""

import sys
import json
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.auth import hash_password
from backend.config.settings import settings


def migrate_passwords(dry_run: bool = False):
    """
    Migrate all plaintext passwords to hashed passwords

    Args:
        dry_run: If True, only show what would be changed
    """
    users_path = Path(settings.users_path)

    if not users_path.exists():
        print(f"❌ Users file not found: {users_path}")
        return

    # Load existing users
    with open(users_path, "r", encoding="utf-8") as f:
        users_data = json.load(f)

    users = users_data.get("users", [])
    total_users = len(users)
    migrated_count = 0
    already_hashed_count = 0

    print(f"Found {total_users} users in {users_path}")
    print("=" * 60)

    for i, user in enumerate(users, 1):
        username = user.get("username", f"user_{i}")
        password = user.get("password", "")
        password_hash = user.get("password_hash", "")

        # Check if already hashed
        if password_hash and (password_hash.startswith("$2b$") or password_hash.startswith("$2a$")):
            print(f"[{i}/{total_users}] {username:<20} ✓ Already hashed")
            already_hashed_count += 1
            continue

        # Check if password is plaintext
        if password and not (password.startswith("$2b$") or password.startswith("$2a$")):
            print(f"[{i}/{total_users}] {username:<20} → Migrating plaintext password")

            if not dry_run:
                # Hash the password
                hashed = hash_password(password)
                user["password_hash"] = hashed

                # Remove old plaintext password for security
                if "password" in user:
                    del user["password"]

            migrated_count += 1
        else:
            print(f"[{i}/{total_users}] {username:<20} ⚠ No password found")

    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total users:      {total_users}")
    print(f"  Already hashed:   {already_hashed_count}")
    print(f"  Needs migration:  {migrated_count}")
    print(f"  Mode:             {'DRY RUN (no changes made)' if dry_run else 'LIVE (changes saved)'}")

    if migrated_count == 0:
        print("\n✓ All passwords are already hashed. No migration needed.")
        return

    if dry_run:
        print(f"\n💡 Run without --dry-run to migrate {migrated_count} passwords")
        return

    # Save updated users
    print(f"\n💾 Saving changes to {users_path}...")

    # Backup original file
    backup_path = users_path.with_suffix(".json.backup")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Backup created: {backup_path}")

    # Save migrated data
    with open(users_path, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Migration complete! {migrated_count} passwords hashed.")
    print(f"\n⚠️ IMPORTANT: Keep the backup file in a secure location!")
    print(f"   Backup: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate plaintext passwords to bcrypt hashed passwords"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )

    args = parser.parse_args()

    print("Password Migration Script")
    print("=" * 60)

    if args.dry_run:
        print("🔍 DRY RUN MODE: No changes will be made")
    else:
        print("⚠️ LIVE MODE: Passwords will be migrated")

    print()

    try:
        migrate_passwords(dry_run=args.dry_run)
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
