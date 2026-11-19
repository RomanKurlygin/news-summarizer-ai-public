"""Формальная проверка черновика и предупреждения о возможных пропусках."""

from backend.agents.base import Agent
from backend.agents.source_checks import (
    attribution_findings, qualification_findings, unattributed_allegation, unsupported_causal_connector,
    unsupported_vehicle_impact,
)
from backend.agents.state import AgentState, FactualFinding, ValidationCode
from backend.agents.validation_entities import (
    glossary_names, is_name_only, new_entity_findings, new_number_findings, possible_omissions,
)
from backend.utils.text_processing import collapse_whitespace, sentence_key, split_sentences

_ERROR_MESSAGES = {
    ValidationCode.MISSING_ORIGINAL_TEXT: "Отсутствует исходный текст новости.",
    ValidationCode.MISSING_SUMMARY: "Резюме отсутствует или пусто.",
    ValidationCode.SENTENCE_COUNT: "Недопустимое число предложений в резюме. Верхний предел — {sentences}.",
    ValidationCode.SUMMARY_TOO_LONG: "Резюме длиннее допустимых {limit} символов.",
    ValidationCode.DUPLICATE_SENTENCE: "В резюме повторяется предложение.",
    ValidationCode.VERBATIM_SENTENCE: "В резюме есть предложение, дословно скопированное из новости.",
    ValidationCode.UNSUPPORTED_NAME: "В резюме есть многословное имя или название, отсутствующее в исходной новости.",
    ValidationCode.UNATTRIBUTED_ALLEGATION: "В резюме деталь из обвинения представлена без указания, кто его выдвинул.",
    ValidationCode.UNSUPPORTED_VEHICLE_IMPACT: "В резюме утверждается наезд, хотя в исходной новости наезд не указан.",
    ValidationCode.UNSUPPORTED_CAUSAL_LINK: "В резюме утверждается причинная связь, которой нет в исходной новости.",
    ValidationCode.UNSUPPORTED_NUMBER: "В резюме изменено числовое значение сопоставимой величины.",
    ValidationCode.LOST_QUALIFICATION: "Резюме представляет предположение или обвинение как установленный факт.",
}
_WARNING_PREFIX = {
    ValidationCode.POTENTIAL_MISSING_NAME: "Возможно, в резюме пропущено имя или название",
    ValidationCode.POTENTIAL_MISSING_DATE: "Возможно, в резюме пропущена дата",
    ValidationCode.POTENTIAL_MISSING_NUMBER: "Возможно, в резюме пропущено число",
}
_REVIEW_MESSAGES = {
    ValidationCode.UNSUPPORTED_ATTRIBUTION: "Источник описывает проверку гипотезы. Сверьте, действительно ли указанный участник утверждал это",
    ValidationCode.UNCONFIRMED_ENTITY: "Нужно сверить новую сущность или вариант её написания",
    ValidationCode.UNCONFIRMED_NUMBER: "Нужно сверить число, единицу измерения или дату",
    ValidationCode.UNCERTAIN_QUALIFICATION: "Нужно проверить, сохранилась ли атрибуция или неопределённость утверждения",
}


class Validator(Agent):
    def __init__(
        self, *, max_summary_chars: int = 600, min_copied_sentence_chars: int = 60,
        check_unsupported_names: bool = False, check_source_claims: bool = False, max_sentences: int = 3,
    ) -> None:
        if min(max_summary_chars, min_copied_sentence_chars) < 1:
            raise ValueError("Пределы длины должны быть положительными.")
        super().__init__(
            "Validator",
            parameters={
                "max_summary_chars": max_summary_chars,
                "min_copied_sentence_chars": min_copied_sentence_chars,
            },
        )
        self.max_summary_chars = max_summary_chars
        self.min_copied_sentence_chars = min_copied_sentence_chars
        self.check_unsupported_names = check_unsupported_names
        self.check_source_claims = check_source_claims
        self.max_sentences = max_sentences

    async def _process(self, state: AgentState) -> AgentState:
        original = collapse_whitespace(state.original_text)
        summary = collapse_whitespace(state.current_summary or "")
        errors: list[ValidationCode] = []
        warnings: dict[ValidationCode, str] = {}
        findings: list[FactualFinding] = []
        if not any(char.isalnum() for char in original):
            errors.append(ValidationCode.MISSING_ORIGINAL_TEXT)
        if not summary:
            errors.append(ValidationCode.MISSING_SUMMARY)
        else:
            sentences = split_sentences(summary)
            keys = [sentence_key(sentence) for sentence in sentences]
            if not 1 <= len(sentences) <= self.max_sentences:
                errors.append(ValidationCode.SENTENCE_COUNT)
            if len(summary) > self.max_summary_chars:
                errors.append(ValidationCode.SUMMARY_TOO_LONG)
            if len(set(keys)) < len(keys):
                errors.append(ValidationCode.DUPLICATE_SENTENCE)

            if original:
                source_sentences = split_sentences(original)
                glossary = glossary_names(state.glossary_context)
                if self.check_unsupported_names:
                    findings.extend(new_entity_findings(original, summary, glossary))
                if self.check_source_claims:
                    findings.extend(new_number_findings(original, summary))
                    findings.extend(qualification_findings(source_sentences, sentences))
                    findings.extend(attribution_findings(source_sentences, sentences))
                    if unattributed_allegation(source_sentences, sentences):
                        findings.append(FactualFinding(
                            code=ValidationCode.UNATTRIBUTED_ALLEGATION, severity="error",
                            category="claim", subject=summary[:160],
                        ))
                    if unsupported_vehicle_impact(original, summary):
                        errors.append(ValidationCode.UNSUPPORTED_VEHICLE_IMPACT)
                    if unsupported_causal_connector(original, summary):
                        errors.append(ValidationCode.UNSUPPORTED_CAUSAL_LINK)
                source_keys = {
                    sentence_key(sentence)
                    for sentence in source_sentences
                    if not is_name_only(sentence, glossary)
                }
                if any(
                    len(key) >= self.min_copied_sentence_chars
                    and key in source_keys and not is_name_only(sentence, glossary)
                    for sentence, key in zip(sentences, keys)
                ):
                    errors.append(ValidationCode.VERBATIM_SENTENCE)
                if source_sentences:
                    for code, labels in possible_omissions(source_sentences[0], summary, glossary).items():
                        names = ", ".join(dict.fromkeys(labels))
                        warnings[code] = (
                            f"{_WARNING_PREFIX[code]} из первого предложения новости ({names})."
                        )

        review_subjects: dict[ValidationCode, list[str]] = {}
        for finding in findings:
            if finding.severity == "error" and finding.code not in errors:
                errors.append(finding.code)
            if finding.severity == "review":
                review_subjects.setdefault(finding.code, []).append(finding.subject)
        for code, subjects in review_subjects.items():
            warnings[code] = f"{_REVIEW_MESSAGES[code]} ({', '.join(dict.fromkeys(subjects))})."

        state.factual_findings = findings
        state.validation_passed = not errors
        state.validation_errors = [
            _ERROR_MESSAGES[code].format(limit=self.max_summary_chars, sentences=self.max_sentences)
            for code in errors
        ]
        state.validation_error_codes = errors
        state.validation_warnings = list(warnings.values())
        state.validation_warning_codes = list(warnings)
        return state
