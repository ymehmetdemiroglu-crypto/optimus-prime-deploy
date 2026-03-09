"""
Automation Agent — Output Validator

Validates execution results against optional JSON Schemas
using jsonschema Draft7Validator.
"""

import logging
from typing import Optional, Dict, Any

from .schemas import ExecutionResult

logger = logging.getLogger(__name__)


class ResultValidator:
    """Validates an ExecutionResult's output against a JSON Schema."""

    @staticmethod
    def validate(result: ExecutionResult, schema: Optional[Dict[str, Any]]) -> ExecutionResult:
        """
        Validate result.output against the given JSON Schema.

        Sets result.validation_passed and result.validation_errors.
        If no schema is provided, validation is skipped (passes by default).
        """
        if schema is None:
            result.validation_passed = True
            return result

        if result.output is None:
            result.validation_passed = False
            result.validation_errors = ["Script produced no parseable JSON output to validate."]
            return result

        try:
            from jsonschema import Draft7Validator

            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(result.output))

            if errors:
                result.validation_passed = False
                result.validation_errors = [
                    f"{e.json_path}: {e.message}" for e in errors
                ]
                logger.warning(
                    f"[Validator] {result.execution_id} failed validation: "
                    f"{len(errors)} error(s)"
                )
            else:
                result.validation_passed = True
                logger.info(f"[Validator] {result.execution_id} passed validation")

        except ImportError:
            logger.error("[Validator] jsonschema package not installed. Skipping validation.")
            result.validation_passed = None
            result.validation_errors = ["jsonschema package not installed"]
        except Exception as e:
            result.validation_passed = False
            result.validation_errors = [f"Validation error: {str(e)}"]
            logger.error(f"[Validator] Unexpected error: {e}")

        return result
