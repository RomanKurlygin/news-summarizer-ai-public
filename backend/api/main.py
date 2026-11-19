"""Каркас HTTP API NewsSummarizer AI."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from backend.api.routes.admin import router as admin_router
from backend.api.routes.auth import router as auth_router
from backend.api.routes.evaluation import router as evaluation_router
from backend.api.routes.health import router as health_router
from backend.api.routes.hitl import router as hitl_router
from backend.api.routes.summarization import router as summarization_router
from backend.api.routes.ui_data import router as ui_data_router
from backend.api.runtime import SummaryRuntime
from backend.core.config import Settings
from backend.core.constants import API_VERSION, REQUEST_ID_HEADER
from backend.core.exceptions import ServiceUnavailableError
from backend.db.session import Database
from backend.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings)
        database = None
        if settings.database_url is not None:
            database = Database(settings.database_url.get_secret_value(), settings)
        application.state.database = database
        logger.info("Application started")
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds, trust_env=False) as http:
                application.state.http = http
                application.state.summary_runtime = SummaryRuntime(settings, database, http) if database else None
                yield
        finally:
            if database is not None:
                await database.close()
            application.state.database = None
            application.state.summary_runtime = None
            application.state.http = None
            logger.info("Application stopped")

    application = FastAPI(
        title=settings.app_name, version=API_VERSION, lifespan=lifespan
    )
    application.state.settings = settings
    application.state.database = None
    application.state.summary_runtime = None
    application.include_router(health_router)
    application.include_router(auth_router)
    application.include_router(admin_router)
    application.include_router(hitl_router)
    application.include_router(summarization_router)
    application.include_router(evaluation_router)
    application.include_router(ui_data_router)

    @application.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(content={"detail": "Request validation failed"}, status_code=422)

    @application.exception_handler(ServiceUnavailableError)
    async def service_unavailable(
        _request: Request, _exc: ServiceUnavailableError
    ) -> JSONResponse:
        return JSONResponse(content={"detail": "Service unavailable"}, status_code=503)

    @application.exception_handler(SQLAlchemyError)
    async def database_unavailable(_request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.warning("Database operation failed (%s)", type(exc).__name__)
        return JSONResponse(content={"detail": "База данных недоступна."}, status_code=503)

    @application.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid4().hex
        request.state.request_id = request_id
        log_context = {"request_id": request_id}
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "Unhandled request failure (%s)", type(exc).__name__, extra=log_context
            )
            response = JSONResponse(content={"detail": "Internal server error"}, status_code=500)

        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info("HTTP response %d", response.status_code, extra=log_context)
        return response

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    @application.get("/")
    async def index() -> dict[str, str]:
        return {"service": settings.app_name, "version": API_VERSION}

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    return application


app = create_app(Settings())
