"""Optional dispatcher sign-in: disabled by default, enabled by ``BACKEND_AUTH_USERS``.

``BACKEND_AUTH_USERS="dispatcher:Иванова И.:<token>;viewer:Табло:<token>"`` — roles are
``dispatcher`` (reads and acts: acknowledgement, replay, demo source) and ``viewer`` (reads
only). Clients send ``Authorization: Bearer <token>``. Tokens are compared in constant time
and only their SHA-256 is kept in memory; they never appear in status or logs. Without the
variable the console behaves as before (a local demo stand without accounts), and
acknowledgements are signed by the generic role name.
"""

import hashlib
import hmac
import os
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["dispatcher", "viewer"]
ROLES = ("dispatcher", "viewer")
MIN_TOKEN_LENGTH = 12
OPEN_PATHS = ("/health/", "/docs", "/redoc", "/openapi.json", "/api/v1/auth")


@dataclass(frozen=True)
class Principal:
    name: str
    role: Role
    authenticated: bool = True

    @property
    def can_act(self) -> bool:
        return self.role == "dispatcher"

    def to_dict(self, enabled: bool) -> dict:
        return {
            "enabled": enabled,
            "authenticated": self.authenticated,
            "name": self.name,
            "role": self.role,
            "can_act": self.can_act,
        }


ANONYMOUS = Principal("Диспетчер", "dispatcher", authenticated=False)


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


@dataclass(frozen=True)
class AuthConfig:
    users: tuple[tuple[bytes, Principal], ...] = field(default_factory=tuple)

    @property
    def enabled(self) -> bool:
        return bool(self.users)

    @classmethod
    def parse(cls, value: str) -> "AuthConfig":
        users = []
        for item in filter(None, (part.strip() for part in value.split(";"))):
            role, separator, rest = item.partition(":")
            name, separator2, token = rest.rpartition(":")
            role, name, token = role.strip(), name.strip(), token.strip()
            if not separator or not separator2 or role not in ROLES or not name:
                raise ValueError("BACKEND_AUTH_USERS entries are role:name:token")
            if len(token) < MIN_TOKEN_LENGTH:
                raise ValueError(f"Auth tokens must have at least {MIN_TOKEN_LENGTH} characters")
            users.append((_digest(token), Principal(name, role)))
        digests = [digest for digest, _ in users]
        if len(set(digests)) != len(digests):
            raise ValueError("Auth tokens must be unique")
        return cls(tuple(users))

    @classmethod
    def from_env(cls) -> "AuthConfig":
        return cls.parse(os.environ.get("BACKEND_AUTH_USERS", ""))

    def principal(self, authorization: str | None) -> Principal | None:
        """The caller, ``ANONYMOUS`` when auth is off, ``None`` for a missing/unknown token."""
        if not self.enabled:
            return ANONYMOUS
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        digest = _digest(token.strip())
        found = None
        for known, principal in self.users:
            # Every entry is compared, so the timing does not reveal which one matched.
            if hmac.compare_digest(known, digest):
                found = principal
        return found


def is_open(path: str) -> bool:
    return not path.startswith("/api/") or path.startswith(OPEN_PATHS)
