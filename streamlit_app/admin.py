"""Управление данными через уже существующие маршруты FastAPI."""

import streamlit as st


def admin_page(client) -> None:
    st.caption("АДМИНИСТРИРОВАНИЕ")
    st.title("База знаний и справочник")
    glossary = st.radio("Раздел данных", ["Справочник", "База знаний"], horizontal=True) == "Справочник"
    resource = "glossary" if glossary else "knowledge-base"
    endpoint = f"/api/v1/admin/{resource}"
    page = st.number_input("Страница записей", min_value=1, step=1, key=f"admin_page_{resource}")
    rows = client.get(endpoint, {"limit": 30, "offset": (page - 1) * 30})
    if notice := st.session_state.pop("admin_notice", None):
        st.success(notice)
    names = {None: "Новая запись"} | {
        row["id"]: f"{row['term' if glossary else 'source_text'][:64]} · {row['id'][:8]}" for row in rows
    }
    row_id = st.selectbox("Запись", list(names), format_func=names.get, key=f"admin_select_{resource}")
    current = client.get(f"{endpoint}/{row_id}") if row_id else {}
    with st.form(f"entry_{resource}_{row_id}"):
        if glossary:
            term = st.text_input("Термин", current.get("term", ""), max_chars=255)
            definition = st.text_area("Определение", current.get("definition", ""), max_chars=16000)
            aliases = st.text_input("Варианты через запятую", ", ".join(current.get("aliases") or []))
            values = {"term": term, "definition": definition,
                      "aliases": [alias.strip() for alias in aliases.split(",") if alias.strip()]}
        else:
            source = st.text_area("Текст новости", current.get("source_text", ""), max_chars=16000)
            target = st.text_area("Эталонное резюме", current.get("target_text", ""), max_chars=16000)
            values = {"source_text": source, "target_text": target}
        left, right = st.columns(2)
        values["category"] = left.text_input("Категория", current.get("category", "general"))
        values["language"] = right.text_input("Код языка", current.get("language", "en"))
        save = st.form_submit_button("Сохранить запись", type="primary")
    if save:
        if row_id:
            client.request("PUT", f"{endpoint}/{row_id}", body=values)
        else:
            client.post(endpoint, values)
        st.session_state.admin_notice = "Запись сохранена."
        st.rerun()
    if row_id:
        confirm = st.checkbox("Подтверждаю удаление выбранной записи", key=f"remove_{row_id}")
        if st.button("Удалить запись", disabled=not confirm):
            client.request("DELETE", f"{endpoint}/{row_id}")
            st.session_state.admin_notice = "Запись удалена."
            st.rerun()
    with st.expander("Импорт CSV"):
        required = "term,definition" if glossary else "source_text,target_text"
        st.caption(f"До 100 записей в UTF-8. Обязательные заголовки — {required}.")
        upload = st.file_uploader("Файл CSV", type=["csv"], key=f"import_{resource}")
        if st.button("Импортировать", disabled=upload is None):
            result = client.request("POST", endpoint + "/import", csv_data=upload.getvalue())
            st.success(f"Импортировано записей: {result['imported']}.")
