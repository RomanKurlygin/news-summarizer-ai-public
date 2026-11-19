"""Русские подписи и адаптация публичных DTO для графиков."""

STATUSES = {
    "draft": "Черновик", "verified": "Автоматически проверено", "needs_review": "Нужна ручная проверка",
    "unchanged": "Правка не изменила результат", "iteration_limit": "Лимит правок исчерпан",
    "failed": "Ошибка обработки", "running": "В работе", "pending": "Ожидает обработки",
    "cancelled": "Отменено", "incomplete_configuration": "Не хватает компонентов", "ok": "Выполнено",
    "approved": "Подтверждено редактором", "rejected": "Отклонено редактором",
}
MODES = {
    "baseline": "Базовый · baseline", "full": "Полный · full", "+dict": "Со справочником · +dict",
    "+rag": "С поиском · +rag", "+agents": "С агентами · +agents", "+ensemble": "Ансамбль · +ensemble",
}
AGENTS = {
    "RAG": "Поиск RAG", "Glossary": "Справочник", "Summarizer": "Создание резюме",
    "Validator": "Проверка правил", "Critic": "Оценка качества", "Editor": "Редактирование",
    "Supervisor": "Итоговый вердикт",
    "Cache": "Повтор из кэша",
}
ROLES = {"user": "Пользователь", "editor": "Редактор", "admin": "Администратор"}
ISSUES = {
    "unsupported_name": "Имя или название отсутствует в источнике",
    "unsupported_number": "Число не подтверждено источником",
    "unconfirmed_number": "Число или дату нужно перепроверить",
    "unconfirmed_entity": "Имя или название нужно перепроверить",
    "unsupported_attribution": "Гипотеза приписана другому участнику как утверждение",
    "lost_qualification": "Потеряна оговорка или степень уверенности",
    "uncertain_qualification": "Атрибуция или оговорка требует проверки",
    "unattributed_allegation": "Обвинение передано без атрибуции",
    "unsupported_causal_link": "Причинная связь не подтверждена",
    "unsupported_vehicle_impact": "Наезд не подтверждён",
    "critic_rejected": "Оценка Critic не прошла порог или содержит критический флаг",
    "critical_factual_error": "Critic отметил возможную существенную ошибку",
    "missing_key_fact": "Critic отметил пропуск ключевого факта",
    "verbatim_sentence": "Предложение скопировано дословно",
    "duplicate_sentence": "Предложение повторяется",
    "sentence_count": "Нарушен предел числа предложений",
    "summary_too_long": "Резюме превышает ограничение длины",
    "potential_missing_name": "Возможно, пропущено имя или название",
    "potential_missing_number": "Возможно, пропущено число",
    "potential_missing_date": "Возможно, пропущена дата",
    "validation_warnings_present": "Валидатор оставил предупреждения",
    "llm_failed": "Сбой генерации",
    "llm_generation_failed": "Сбой генерации модели",
    "llm_timeout": "Превышено время ожидания модели",
    "language_mismatch": "Язык ответа модели не совпал с выбранным",
    "sentence_limit": "Превышен выбранный предел предложений",
    "rag_failed": "Не удалось выполнить поиск",
}


def issue(code: str) -> str:
    return ISSUES.get(code, f"Код проверки · {code}")


def archive_rows(data: dict) -> list[dict]:
    return [row | {"label": MODES[row["mode"]], "success_rate": row["successful"] / row["processed"]}
            for row in data["rows"]]


def history_rows(entries: list[dict]) -> list[dict]:
    rows = []
    for entry in entries:
        aggregate = entry.get("aggregate") or {}
        metrics = aggregate.get("metrics") or entry.get("metrics") or {}
        if not metrics:
            continue
        total = entry["sample_size"]
        count = aggregate.get("success_count", entry.get("scored_count"))
        checked = aggregate.get("validator_checked_count")
        rows.append({
            **metrics, "mode": entry["mode"], "processed": total, "successful": count,
            "label": f"{entry['model_name']} · {MODES.get(entry['mode'], entry['mode'])} · {total} · {entry['id'][:8]}",
            "success_rate": count / total if count is not None and total else None,
            "mean_seconds": aggregate.get("mean_seconds"),
            "verified_rate": aggregate["verified_count"] / total if checked else None,
            "needs_review_rate": aggregate["needs_review_count"] / total if checked else None,
        })
    return rows
