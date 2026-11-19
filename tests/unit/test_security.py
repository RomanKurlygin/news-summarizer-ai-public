from secrets import token_urlsafe
from uuid import uuid4

import pytest

from backend.core.config import Settings
from backend.core.exceptions import ServiceUnavailableError
from backend.utils.security import (
    create_access_token,
    decode_access_token,
    get_signing_key,
    verify_password,
)


@pytest.mark.parametrize(
    ("algorithm", "length"), [("HS256", 31), ("HS384", 47), ("HS512", 63)]
)
def test_short_hmac_keys_are_rejected_in_production(algorithm, length):
    settings = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret_key="x" * length,
        jwt_algorithm=algorithm,
    )
    with pytest.raises(ServiceUnavailableError, match="too short"):
        get_signing_key(settings)


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
def test_token_algorithm_is_configurable(algorithm):
    settings = Settings(_env_file=None, jwt_secret_key=token_urlsafe(64), jwt_algorithm=algorithm)
    user_id = uuid4()
    token = create_access_token(user_id, settings, get_signing_key(settings))
    assert decode_access_token(token, settings) == user_id


def test_password_verification_handles_bcrypt_byte_limit_and_invalid_stored_hash():
    assert verify_password("я" * 37, None) is False
    assert verify_password("fixture-password", "malformed-hash") is False
