"""Общие исключения для прикладных компонентов."""


class AppError(Exception):
    """Базовое исключение приложения."""


class ServiceUnavailableError(AppError):
    """Зависимый сервис временно недоступен."""
