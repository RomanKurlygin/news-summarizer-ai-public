import json
from unittest.mock import AsyncMock

import pytest

from backend.agents.state import AgentState, ValidationCode
from backend.agents.supervisor import Supervisor
from backend.agents.validator import Validator
from backend.core.config import Settings
from backend.db.init_db import create_test_schema
from backend.db.reasoning import ReasoningLogStore
from backend.db.session import Database
from backend.utils.text_processing import split_sentences

pytestmark = pytest.mark.asyncio

NEWS = "Вдоль набережной открыли парк. Жители уже гуляют по новым дорожкам."
PARAPHRASE = "На набережной начал работать парк для прогулок."
LONG_SENTENCE = (
    "Вдоль набережной после завершения благоустройства открыли новый общедоступный парк."
)


@pytest.mark.parametrize(("source", "draft"), [
    ("В Казани открыли библиотеку.", "A library opened in Kazan."),
    ("A library opened in Kazan.", "В Казани начала работать библиотека."),
    ("В Казани открыли библиотеку.", "Казань получила новую библиотеку."),
    ("В Москве открыли библиотеку.", "A library opened in Moscow."),
    ("В Санкт-Петербурге открыли библиотеку.", "A library opened in Saint Petersburg."),
    ("A library opened in Saint Petersburg.", "В Санкт-Петербурге начала работать библиотека."),
    ("В Новосибирске начали проект.", "The project started in Novosibirsk."),
    ("Анна Соколова открыла библиотеку.", "Anna Sokolova opened the library."),
])
async def test_validator_recognizes_supported_places_and_transliterations(source, draft):
    result = await Validator(check_unsupported_names=True).process(AgentState(original_text=source, current_summary=draft))
    assert not result.factual_findings and result.validation_passed


@pytest.mark.parametrize("place", ["Moscow", "Saint Petersburg", "London"])
async def test_validator_does_not_exempt_new_english_places(place):
    result = await Validator(check_unsupported_names=True).process(AgentState(
        original_text="В Казани открыли новую библиотеку.", current_summary=f"The library opened in {place}.",
    ))
    assert any(finding.category == "place" and finding.subject == place for finding in result.factual_findings)
    assert result.validation_passed is False or result.factual_review_required


@pytest.mark.parametrize(("source", "draft", "code"), [
    (
        "Police asked whether witnesses saw a courier or a truck before the incident.",
        "A courier was hit by a truck, and police sought witnesses.",
        ValidationCode.UNSUPPORTED_VEHICLE_IMPACT,
    ),
    (
        "The former employee accused the mayor of running a 12-month spying campaign.",
        "The mayor ran a 12-month spying campaign.",
        ValidationCode.UNATTRIBUTED_ALLEGATION,
    ),
    (
        "The former employee accused the mayor of running a 12-month spying campaign.",
        "The former employee alleged a 12-month spying campaign by the mayor.",
        None,
    ),
    (
        "Police confirmed that a delivery van collided with a jogger before noon.",
        "A jogger was hit by a delivery van before noon.",
        None,
    ),
    (
        "Inspectors visited a factory on Monday. Its director resigned the next day.",
        "Inspectors visited a factory, prompting its director to resign.",
        ValidationCode.UNSUPPORTED_CAUSAL_LINK,
    ),
    (
        "An electrical failure caused staff to leave the museum before closing time.",
        "An electrical fault occurred, prompting staff to leave the museum.",
        None,
    ),
])
async def test_validator_source_checks_keep_causation_and_claims_grounded(source, draft, code):
    result = await Validator(check_source_claims=True).process(AgentState(original_text=source, current_summary=draft))
    if code is None:
        assert not result.validation_error_codes
    else:
        assert code in result.validation_error_codes
    assert result.validation_passed is (code is None)


@pytest.mark.parametrize(('source', 'draft', 'code', 'severity', 'category'), [
    pytest.param('Konta hit 11 aces and beat the American wildcard 6–3, 6–1.',
                 'Johanna Konta beat the wildcard 6-3, 6-1 and hit 11 aces.',
                 ValidationCode.UNCONFIRMED_ENTITY, 'review', 'person', id='xsum-36852433'),
    ('A festival opened in Lisbon after the renovations.', 'The festival opened in Porto.',
     ValidationCode.UNCONFIRMED_ENTITY, 'review', 'place'),
    ('Konta won the match after a long rally.', 'Williams won the match.',
     ValidationCode.UNCONFIRMED_ENTITY, 'review', 'person'),
    ('Teams found three parcels in the warehouse.', 'Teams found 9 parcels.',
     ValidationCode.UNSUPPORTED_NUMBER, 'error', 'number'),
    ('Police accused the contractor of stealing public funds from the hospital.',
     'The contractor stole public funds from the hospital.',
     ValidationCode.LOST_QUALIFICATION, 'error', 'claim'),
    ('Police accused the contractor of stealing public funds from the hospital. The hospital closed.',
     'The hospital reportedly closed, and the contractor stole public funds from the hospital.',
     ValidationCode.LOST_QUALIFICATION, 'error', 'claim'),
    ('Следователи предполагают, что подрядчик присвоил деньги на ремонт школы.',
     'Подрядчик присвоил деньги на ремонт школы.',
     ValidationCode.LOST_QUALIFICATION, 'error', 'claim'),
    ('According to the contractor, inspectors closed the hospital after floods.',
     'Inspectors closed the hospital after floods.',
     ValidationCode.UNCERTAIN_QUALIFICATION, 'review', 'claim'),
])
async def test_factual_findings_are_structured(source, draft, code, severity, category):
    result = await Validator(check_unsupported_names=True, check_source_claims=True).process(
        AgentState(original_text=source, current_summary=draft)
    )
    assert result.original_text == source and result.current_summary == draft
    finding = next(item for item in result.factual_findings if item.code == code)
    assert (finding.severity, finding.category) == (severity, category)
    assert result.validation_passed is (severity == 'review')
    assert result.factual_review_required is (severity == 'review')


@pytest.mark.parametrize(('source', 'draft'), [
    ('Police accused the contractor of stealing public funds from the hospital.',
     'According to police, the contractor may have stolen public funds from the hospital.'),
    ('Следователи предполагают, что подрядчик присвоил деньги на ремонт школы.',
     'По версии следователей, подрядчик присвоил деньги на ремонт школы.'),
    ('An engineer reportedly saw cracks, but inspectors confirmed the bridge reopened on Monday.',
     'The bridge reopened on Monday.'),
    ('The World Health Organization published a report about the local clinic.',
     'A report about the local clinic was published by WHO.'),
    ('The DR Congo government announced a new investigation about missing people.',
     'A new investigation is underway in the Democratic Republic of Congo.'),
    ('Москва стала площадкой для нового фестиваля.', 'Фестиваль прошёл в Москве.'),
    ('The charity said 3,000 people attended, and three awards were presented.',
     'The event welcomed 3000 people and included 3 awards.'),
])
async def test_factual_checks_accept_supported_variants(source, draft):
    result = await Validator(check_unsupported_names=True, check_source_claims=True).process(
        AgentState(original_text=source, current_summary=draft)
    )
    assert result.validation_passed is True
    assert not result.factual_findings


@pytest.mark.parametrize("summary", [
    PARAPHRASE,
    "На набережной заработал парк. Жители гуляют по дорожкам. Вдоль реки стало удобнее ходить.",
])
async def test_validator_accepts_one_and_three_sentences(summary):
    result = await Validator().process(AgentState(original_text=NEWS, current_summary=summary))

    assert result.validation_passed is True
    assert result.validation_errors == result.validation_warnings == []
    assert result.validation_error_codes == result.validation_warning_codes == []


@pytest.mark.parametrize("summary", [None, "", " \n\t "])
async def test_empty_or_missing_summary_is_an_error(summary):
    result = await Validator().process(AgentState(original_text=NEWS, current_summary=summary))

    assert result.validation_passed is False
    assert result.validation_error_codes == [ValidationCode.MISSING_SUMMARY]
    assert result.validation_errors == ["Резюме отсутствует или пусто."]


@pytest.mark.parametrize("input_fields", [{}, {"original_text": None}, {"original_text": " \n"}])
async def test_absent_source_text_is_an_error(input_fields):
    result = await Validator().process(AgentState(current_summary=PARAPHRASE, **input_fields))

    assert result.validation_passed is False
    assert result.validation_error_codes == [ValidationCode.MISSING_ORIGINAL_TEXT]


@pytest.mark.parametrize("summary", ["?!", "Открыли парк. Горожане пришли. Они гуляли. Включили свет."])
async def test_sentence_count_rejects_zero_and_four(summary):
    result = await Validator().process(AgentState(original_text=NEWS, current_summary=summary))

    assert result.validation_passed is False
    assert ValidationCode.SENTENCE_COUNT in result.validation_error_codes


@pytest.mark.parametrize("summary", [
    "Открылся парк. Открылся парк.",
    "Открылся парк. ОТКРЫЛСЯ ПАРК.",
    "Парк, где можно гулять. Парк , где\tможно  гулять.",
])
async def test_duplicate_sentences_are_normalized(summary):
    result = await Validator().process(AgentState(original_text=NEWS, current_summary=summary))

    assert result.validation_passed is False
    assert result.validation_error_codes == [ValidationCode.DUPLICATE_SENTENCE]


async def test_repeated_word_in_one_sentence_is_allowed():
    result = await Validator().process(
        AgentState(original_text=NEWS, current_summary="Парк стал больше, и парк стал удобнее.")
    )

    assert result.validation_passed is True


async def test_length_limit_is_configurable_and_uses_normalized_whitespace():
    state = AgentState(original_text=NEWS, current_summary=" \tОткрыт   парк.  ")
    normalized_length = len("Открыт парк.")

    assert (await Validator(max_summary_chars=normalized_length).process(state)).validation_passed
    result = await Validator(max_summary_chars=normalized_length - 1).process(state)
    assert result.validation_error_codes == [ValidationCode.SUMMARY_TOO_LONG]
    with pytest.raises(ValueError, match="Пределы"):
        Validator(min_copied_sentence_chars=0)


async def test_long_verbatim_sentence_is_detected_despite_case_and_whitespace():
    state = AgentState(
        original_text=LONG_SENTENCE + " Гости пришли вечером.",
        current_summary=LONG_SENTENCE.upper().replace(" ", " \n  "),
    )
    result = await Validator().process(state)

    assert result.validation_passed is False
    assert result.validation_error_codes == [ValidationCode.VERBATIM_SENTENCE]
    relaxed = await Validator(min_copied_sentence_chars=150).process(state)
    assert relaxed.validation_passed is True


@pytest.mark.parametrize("phrase", [
    "«Решение принято».",
    "ООО «Национальный исследовательский институт городской инфраструктуры».",
])
async def test_short_quotes_and_organization_names_are_allowed(phrase):
    result = await Validator().process(AgentState(
        original_text=phrase, current_summary=phrase.lower(),
    ))

    assert result.validation_passed is True
    assert result.validation_error_codes == []


@pytest.mark.parametrize(("text", "sentences"), [
    (
        "В г. Казани выступил А. С. Пушкин и т. д. Событие продолжилось вечером.",
        ["В г. Казани выступил А. С. Пушкин и т. д.", "Событие продолжилось вечером."],
    ),
    (
        "В 2026 г. запланировали выставку. Её открыли 21.04.2026.",
        ["В 2026 г. запланировали выставку.", "Её открыли 21.04.2026."],
    ),
    (
        "Объект открыли в 2026 г. Парк работает.",
        ["Объект открыли в 2026 г.", "Парк работает."],
    ),
    (
        "Dr. J. R. Smith met Mr. Jones. They measured 3.14 metres. The meeting ended.",
        ["Dr. J. R. Smith met Mr. Jones.", "They measured 3.14 metres.", "The meeting ended."],
    ),
    (
        "Mr. Lane arrived on Feb. 3, 2026. Dr. Lee returned.",
        ["Mr. Lane arrived on Feb. 3, 2026.", "Dr. Lee returned."],
    ),
])
async def test_russian_and_english_sentence_boundaries(text, sentences):
    assert split_sentences(text) == sentences
    assert (await Validator().process(
        AgentState(original_text=NEWS, current_summary=text)
    )).validation_passed


async def test_detects_names_organizations_dates_and_numbers_as_warnings():
    state = AgentState(
        original_text="Иван Петров сообщил, что NASA запустит проект 15 марта 2026 года с 25 участниками.",
        current_summary="Запланирован запуск проекта.",
    )
    result = await Validator().process(state)

    assert result.validation_passed is True
    assert result.validation_errors == []
    assert result.validation_warning_codes == [
        ValidationCode.POTENTIAL_MISSING_NAME,
        ValidationCode.POTENTIAL_MISSING_DATE,
        ValidationCode.POTENTIAL_MISSING_NUMBER,
    ]
    assert "Иван Петров, NASA" in result.validation_warnings[0]
    assert "15 марта 2026" in result.validation_warnings[1]
    assert "25" in result.validation_warnings[2]


@pytest.mark.parametrize(("news", "summary", "organization"), [
    (
        "Компания «Ромашка» объявила о запуске проекта.",
        "Объявили о запуске проекта.",
        "Ромашка",
    ),
    (
        "OpenAI announced a new research programme.",
        "A new research programme was announced.",
        "OpenAI",
    ),
])
async def test_detects_organizations_without_legal_suffix(news, summary, organization):
    result = await Validator().process(AgentState(original_text=news, current_summary=summary))

    assert result.validation_warning_codes == [ValidationCode.POTENTIAL_MISSING_NAME]
    assert organization in result.validation_warnings[0]


async def test_glossary_recognizes_names_beyond_capitalization_heuristics():
    result = await Validator().process(AgentState(
        original_text="На реке заработал гидропост для наблюдения за паводком.",
        current_summary="На реке начали следить за паводком.",
        glossary_context=["гидропост — пункт наблюдения"],
    ))

    assert result.validation_passed is True
    assert result.validation_warning_codes == [ValidationCode.POTENTIAL_MISSING_NAME]
    assert "гидропост" in result.validation_warnings[0]
    assert "пункт наблюдения" not in result.validation_warnings[0]


@pytest.mark.parametrize(("news", "summary", "glossary"), [
    (
        "ООО «Северная Звезда» открыло пункт помощи.",
        "СЗ запустила новый пункт помощи.",
        ["ООО «Северная Звезда» | СЗ | Severnaya Zvezda"],
    ),
    (
        "АО «Северная Звезда» открыло пункт помощи.",
        "Северная звезда запустила новый пункт помощи.",
        [],
    ),
    (
        "Acme Labs, Inc. opened a new research centre.",
        "A research centre was opened by ACME labs.",
        [],
    ),
    (
        "Проект представила Алёна Петрова.",
        "Алена Петрова выступила с презентацией проекта.",
        [],
    ),
])
async def test_matches_variants_of_names_and_organizations(news, summary, glossary):
    result = await Validator().process(AgentState(
        original_text=news, current_summary=summary, glossary_context=glossary,
    ))

    assert result.validation_passed is True
    assert result.validation_warnings == []


@pytest.mark.parametrize(("news", "summary"), [
    (
        "21.04.2026 открыли парк и отметили рост показателя на 1.5%.",
        "Парк заработал 21 апреля 2026 года, а показатель поднялся на 1,50%.",
    ),
    (
        "The park opened on April 21, 2026, and the budget increased by 1.5%.",
        "On 2026-04-21 the park opened and the budget rose by 1.50%.",
    ),
    (
        "The park opened on 04/21/2026.",
        "The opening took place on 21 April 2026.",
    ),
])
async def test_matches_equivalent_dates_and_decimal_numbers(news, summary):
    result = await Validator().process(AgentState(original_text=news, current_summary=summary))

    assert result.validation_warnings == []
    assert result.validation_passed is True


async def test_secondary_entities_outside_lead_need_not_be_repeated():
    result = await Validator().process(AgentState(
        original_text=NEWS + " На пресс-конференции 14 марта выступил Иван Сидоров.",
        current_summary=PARAPHRASE,
    ))

    assert result.validation_passed is True
    assert result.validation_warnings == []


async def test_state_is_updated_without_altering_source_or_draft():
    state = AgentState(
        original_text="  " + NEWS.replace(" ", "\t") + "\n",
        current_summary=" \t" + PARAPHRASE,
        validation_passed=False,
        validation_errors=["Прежняя ошибка"],
        validation_error_codes=[ValidationCode.SENTENCE_COUNT],
    )
    snapshot = state.model_dump()
    result = await Validator().process(state)

    assert state.model_dump() == snapshot
    assert result.original_text == snapshot["original_text"]
    assert result.current_summary == snapshot["current_summary"]
    assert result.validation_passed is True
    assert result.validation_errors == result.validation_error_codes == []


@pytest.mark.parametrize(("summary", "passes"), [
    (PARAPHRASE, True),
    ("В городе открыли парк. В городе открыли парк.", False),
])
async def test_supervisor_runs_real_validator_when_critic_is_absent(fake_llm, summary, passes):
    recorder = AsyncMock()
    supervisor = Supervisor(lambda: fake_llm(summary), recorder)
    result = await supervisor.run(AgentState(original_text=NEWS))

    assert result.current_summary == summary
    assert result.validation_passed is passes
    assert result.critic_passed is None and result.critic_score is None
    assert result.execution_status == "incomplete_configuration"
    assert [step.agent_name for step in result.execution_history] == [
        "Summarizer", "Validator", "Supervisor",
    ]
    recorder.assert_awaited_once_with(result, "full", None)


async def test_validator_reasoning_log_persists_verdict_counts_and_codes(tmp_path, fake_llm):
    url = f"sqlite+aiosqlite:///{tmp_path / 'validator.sqlite3'}"
    database = Database(url, Settings(_env_file=None))
    marker = "synthetic-confidential-value"
    try:
        await create_test_schema(database.engine)
        store = ReasoningLogStore(database.sessions)
        supervisor = Supervisor(
            lambda: fake_llm("Готовится проект. Готовится проект."), store.save_run
        )
        result = await supervisor.run(AgentState(
            original_text=f"Проект от {marker} объявлен 15 марта 2026 года.",
            glossary_context=[marker],
        ))
        records = await store.list_for_request(result.request_id)
    finally:
        await database.close()

    step = records[1]
    assert step.agent_name == "Validator"
    assert step.request_id == result.request_id
    assert step.started_at <= step.finished_at
    assert step.parameters["duration_ms"] >= 0
    assert step.status == "ok"
    output = json.loads(step.output_summary)
    assert output["validation_passed"] is False
    assert output["validation_errors"] == 1
    assert output["validation_warnings"] == 2
    assert step.errors == ["duplicate_sentence"]
    assert step.warnings == ["potential_missing_name", "potential_missing_date"]
    assert marker in result.validation_warnings[0]
    assert marker not in json.dumps([step.input_summary, output, step.errors, step.warnings])


@pytest.mark.parametrize(("example", "news", "summary", "passes", "errors", "warnings"), [
    ("A", NEWS, PARAPHRASE, True, [], []),
    (
        "B", NEWS, "Открылся парк. В нём есть дорожки. Жители пришли. Они гуляют.",
        False, [ValidationCode.SENTENCE_COUNT], [],
    ),
    (
        "C", "15 марта 2026 года на набережной открыли парк. Вдоль реки проложили дорожки.",
        "На набережной заработал парк с прогулочными дорожками.",
        True, [], [ValidationCode.POTENTIAL_MISSING_DATE],
    ),
])
async def test_validator_demo(example, news, summary, passes, errors, warnings):
    result = await Validator().process(AgentState(original_text=news, current_summary=summary))
    assert result.validation_passed is passes
    assert result.validation_error_codes == errors
    assert result.validation_warning_codes == warnings

    print(json.dumps({
        "example": example,
        **result.model_dump(mode="json", include={
            "original_text", "current_summary", "validation_passed",
            "validation_errors", "validation_warnings",
        }),
    }, ensure_ascii=False))
