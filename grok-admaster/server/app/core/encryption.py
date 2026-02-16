"""
Encryption utilities for protecting sensitive data at rest.

Uses Fernet symmetric encryption from the cryptography library.
The encryption key is derived from the SECRET_KEY in settings.
"""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy.types import TypeDecorator, String
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cipher instance (initialized lazily)
_cipher: Optional[Fernet] = None
_SALT = b"grok-admaster-encryption-salt-v1"  # Static salt for key derivation


def _get_cipher() -> Fernet:
    """Get or create the Fernet cipher instance."""
    global _cipher

    if _cipher is None:
        from app.core.config import settings

        # Derive a Fernet-compatible key from SECRET_KEY
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=_SALT,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(settings.SECRET_KEY.encode()))
        _cipher = Fernet(key)
        logger.info("Encryption cipher initialized")

    return _cipher


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string value.

    Args:
        plaintext: The value to encrypt

    Returns:
        Base64-encoded encrypted value
    """
    if not plaintext:
        return plaintext

    cipher = _get_cipher()
    encrypted_bytes = cipher.encrypt(plaintext.encode())
    return encrypted_bytes.decode()


def decrypt_value(encrypted: str) -> str:
    """Decrypt an encrypted string value.

    Args:
        encrypted: Base64-encoded encrypted value

    Returns:
        Decrypted plaintext string
    """
    if not encrypted:
        return encrypted

    cipher = _get_cipher()
    decrypted_bytes = cipher.decrypt(encrypted.encode())
    return decrypted_bytes.decode()


class EncryptedString(TypeDecorator):
    """SQLAlchemy column type that automatically encrypts/decrypts string values.

    Usage:
        class MyModel(Base):
            secret_field = Column(EncryptedString(255))

    The value is encrypted before storage and decrypted when retrieved.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Encrypt value before storing in database."""
        if value is None:
            return value

        try:
            return encrypt_value(value)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError("Failed to encrypt sensitive data") from e

    def process_result_value(self, value, dialect):
        """Decrypt value after retrieving from database."""
        if value is None:
            return value

        try:
            return decrypt_value(value)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt sensitive data") from e


class PartiallyRedactedString(TypeDecorator):
    """Column type that shows only partial value for debugging.

    Use this for non-critical but sensitive fields like client IDs.
    """

    impl = String
    cache_ok = True

    def process_result_value(self, value, dialect):
        """Return the full value (no encryption, just type marker)."""
        return value

    def redacted_repr(self, value: Optional[str]) -> str:
        """Get a redacted representation for logging."""
        if not value or len(value) <= 4:
            return "***"
        return f"{value[:4]}...***"
