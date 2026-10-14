"""
One Dhan access token shared by every bot on this machine.

The equity client and the FnO agent log in to the same Dhan account. Each TOTP login issues a new
token and invalidates the previous one, so two bots logging in separately keep killing each other's
token ("DH-906 Invalid Token"). Instead, both keep the current token in one file:

- On start, use the token in the file if there is one (no new login).
- When Dhan rejects the token, re-read the file first: if the other bot already logged in, adopt
  its token; only log in again if the file still holds the rejected token.
- A lock file makes sure only one bot logs in at a time.
"""

import fcntl
import logging
import os
import time
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)

TOKEN_FILE = os.environ.get("DHAN_TOKEN_FILE") or os.path.join(os.path.expanduser("~"), ".dhan_token")
MAX_LOGIN_RETRIES = 5


def is_token_error(status_code: int, body) -> bool:
    """Dhan signals an expired/replaced token with 401/403, DH-901, or DH-906 'Invalid Token'."""
    text = str(body).lower()
    return status_code in (401, 403) or "dh-901" in text or "invalid token" in text or "unauthorized" in text


def read_token(path: Optional[str] = None) -> str:
    path = path or TOKEN_FILE
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _write_token(token: str, path: str):
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    os.replace(tmp, path)


@contextmanager
def _locked(path: str):
    fd = os.open(f"{path}.lock", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _execute_with_retries(login: Callable[[], Optional[str]]) -> Optional[str]:
    """
    Executes the login callback with a strict cap of MAX_LOGIN_RETRIES (5) 
    and a 5-second delay to prevent 24-hour broker account lockouts.
    """
    attempts = 0
    while attempts < MAX_LOGIN_RETRIES:
        attempts += 1
        try:
            token = login()
            if token:
                logger.info(f"Dhan authentication successful on attempt {attempts}.")
                return token
            logger.warning(f"Login attempt {attempts}/{MAX_LOGIN_RETRIES} returned an empty token.")
        except Exception as e:
            logger.error(f"Login attempt {attempts}/{MAX_LOGIN_RETRIES} encountered an exception: {e}")
            
        if attempts < MAX_LOGIN_RETRIES:
            logger.info("Retrying token generation in 5 seconds...")
            time.sleep(5)
            
    error_msg = f"CRITICAL: Maximum retry limit ({MAX_LOGIN_RETRIES}) reached for Dhan authentication. Aborting to prevent 24-hour account lockout."
    logger.critical(error_msg)
    return None


def fresh_token(rejected: str, login: Callable[[], Optional[str]], path: Optional[str] = None) -> Optional[str]:
    """
    Return a usable token after `rejected` failed (pass "" when there is no token yet).

    Adopts a newer token from the shared file if another bot already logged in; otherwise calls
    `login()` via the retry wrapper and shares the result.
    """
    path = path or TOKEN_FILE
    with _locked(path):
        shared = read_token(path)
        if shared and shared != rejected:
            logger.info("Using the Dhan token shared by the other bot (no new login needed).")
            return shared
        token = _execute_with_retries(login)
        if token:
            _write_token(token, path)
        return token


def force_login(login: Callable[[], Optional[str]], path: Optional[str] = None) -> Optional[str]:
    """Scheduled daily refresh: always log in with retries, then share the new token."""
    path = path or TOKEN_FILE
    with _locked(path):
        token = _execute_with_retries(login)
        if token:
            _write_token(token, path)
        return token
