"""Схемы публичного API аутентификации."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, StringConstraints, field_validator

from backend.utils.security import BCRYPT_MAX_PASSWORD_BYTES

Username = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, min_length=3, max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    ),
]


class RegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    username: Username
    email: EmailStr
    password: SecretStr = Field(min_length=15, max_length=BCRYPT_MAX_PASSWORD_BYTES)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, email: str) -> str:
        return email.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        password = value.get_secret_value()
        if not password.strip():
            raise ValueError("Password must not be blank")
        if len(password.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES:
            raise ValueError("Password must fit within 72 bytes in UTF-8")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    username: Username
    password: SecretStr = Field(min_length=1, max_length=BCRYPT_MAX_PASSWORD_BYTES)


class UserProfile(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    email: str
    role: Literal["user", "editor", "admin"]
    is_active: bool


class AccessToken(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
