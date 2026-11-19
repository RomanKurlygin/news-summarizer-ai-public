"""Авторские исходные тексты для живого прогона."""

LIBRARY_NEWS = (
    "В Казани открыли новую городскую библиотеку. В ней оборудовали 50 рабочих мест, компьютерный класс и зал "
    "для проведения лекций. Библиотека будет работать ежедневно с 9:00 до 21:00. Посетители смогут бесплатно "
    "пользоваться электронными каталогами и читать книги в специально оборудованных зонах."
)

SCENARIOS = {
    "A · Короткая новость": (
        LIBRARY_NEWS,
        "baseline",
        2,
    ),
    "B · Новость для RAG и агентов": (
        LIBRARY_NEWS,
        "full",
        2,
    ),
    "C · Новость с неподтверждённой версией": (
        "The Harbour Authority said crews recovered items from a flooded warehouse on Thursday. The authority "
        "said the items included empty boxes and damaged tools. The Workers' Union says inspectors must check "
        "whether stolen medical supplies have been stored in the warehouse. Police confirmed they opened "
        "an investigation and said they had not identified any stolen supplies.",
        "full",
        3,
    ),
}
