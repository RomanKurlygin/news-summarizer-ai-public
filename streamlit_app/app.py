"""Запускать из корня проекта через python -m streamlit run streamlit_app/app.py."""

import os

import streamlit as st

from streamlit_app.admin import admin_page
from streamlit_app.api_client import APIClient, APIError
from streamlit_app.presentation import ROLES
from streamlit_app.views import (
    AGENTS_PAGE, COMPARE_PAGE, DATA_PAGE, HISTORY_PAGE, MONITOR_PAGE, REVIEW_PAGE, SUMMARY_PAGE,
    agents_page, comparison_page, history_page, monitor_page, review_page, summarize_page,
)

_BASE = os.getenv("NEWS_API_URL", "http://127.0.0.1:8000")
_THEME = """
<style>
.stMainBlockContainer { max-width: 1120px; padding-top: 2.25rem; }
[data-testid="stSidebar"] { border-right: 1px solid #DCE7EB; }
[data-testid="stAppDeployButton"], .stAppDeployButton { display: none; }
[data-testid="stMetric"] { padding: .65rem .8rem; border: 1px solid #DCE7EB;
  border-radius: .45rem; background: #FFFFFF; min-height: 4.7rem; }
[data-testid="stFileUploaderDropzoneInstructions"] > div { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"]::after { content: "Перетащите файл сюда"; }
[data-testid="stFileUploaderDropzone"] button { font-size: 0; }
[data-testid="stFileUploaderDropzone"] button::after { content: "Выбрать файлы"; font-size: .875rem; }
</style>
"""


def logout(notice: str | None = None) -> None:
    st.session_state.clear()
    if notice:
        st.session_state.login_notice = notice


def login_screen() -> None:
    st.caption("РАБОЧЕЕ ПРОСТРАНСТВО")
    st.title("NewsSummarizer AI")
    st.write("Войдите, чтобы создавать резюме, проверять факты и открывать историю запросов.")
    if notice := st.session_state.pop("login_notice", None):
        st.warning(notice)
    login, register = st.tabs(["Вход", "Регистрация"])
    with login:
        with st.form("login", clear_on_submit=True):
            username = st.text_input("Имя пользователя")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", type="primary")
        if submitted:
            try:
                response = APIClient(_BASE).post("/api/v1/auth/login", {"username": username, "password": password})
            except APIError as error:
                st.error("Неверное имя пользователя или пароль." if error.status == 401 else str(error))
            else:
                st.session_state.token = response["access_token"]
                st.rerun()
    with register:
        with st.form("register", clear_on_submit=True):
            name = st.text_input("Имя для регистрации")
            email = st.text_input("Электронная почта")
            secret = st.text_input("Пароль от 15 символов", type="password")
            submitted = st.form_submit_button("Зарегистрироваться")
        if submitted:
            try:
                APIClient(_BASE).post("/api/v1/auth/register", {"username": name, "email": email, "password": secret})
            except APIError as error:
                st.error(str(error))
            else:
                st.success("Учётная запись создана. Перейдите во вкладку «Вход».")


def main() -> None:
    st.set_page_config(page_title="NewsSummarizer AI", layout="wide")
    st.html(_THEME)
    if not (token := st.session_state.get("token")):
        login_screen()
        return
    client = APIClient(_BASE, token)
    try:
        profile = client.get("/api/v1/auth/me")
        st.session_state.setdefault("preferences", {})
        pages = {SUMMARY_PAGE: summarize_page, AGENTS_PAGE: agents_page,
                 COMPARE_PAGE: comparison_page, HISTORY_PAGE: history_page}
        if profile["role"] in {"editor", "admin"}:
            pages[REVIEW_PAGE] = review_page
        if profile["role"] == "admin":
            pages[DATA_PAGE] = admin_page
        pages[MONITOR_PAGE] = monitor_page
        if st.session_state.get("navigation") not in pages:
            st.session_state.navigation = SUMMARY_PAGE
        with st.sidebar:
            st.subheader("NewsSummarizer AI")
            st.caption("Новостные резюме и проверка качества")
            st.divider()
            page = st.radio("Раздел", list(pages), key="navigation")
            st.divider()
            st.text(profile["username"])
            st.caption(ROLES.get(profile["role"], "Пользователь"))
            st.button("Выйти", on_click=logout)
        pages[page](client)
    except APIError as error:
        if error.status == 401:
            logout("Сеанс закончился. Войдите повторно.")
            st.rerun()
        st.error(str(error))
        st.button("Повторить запрос")
        st.button("Завершить сеанс", on_click=logout)


if __name__ == "__main__":
    main()
