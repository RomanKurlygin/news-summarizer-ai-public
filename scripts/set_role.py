"""Локально назначить роль уже зарегистрированному пользователю."""

import argparse
import asyncio

from sqlalchemy import select

from backend.core.config import Settings
from backend.db.models import User
from backend.db.session import Database


async def assign(username: str, role: str) -> None:
    settings = Settings()
    if not settings.database_url:
        raise ValueError("Задайте DATABASE_URL.")
    database = Database(settings.database_url.get_secret_value(), settings)
    try:
        async with database.sessions.begin() as session:
            user = await session.scalar(select(User).where(User.username == username.strip().casefold()))
            if user is None:
                raise ValueError("Пользователь не найден.")
            user.role = role
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    parser.add_argument("role", choices=("user", "editor", "admin"))
    args = parser.parse_args()
    try:
        asyncio.run(assign(args.username, args.role))
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(f"Роль назначена: {args.role}.")


if __name__ == "__main__":
    main()
