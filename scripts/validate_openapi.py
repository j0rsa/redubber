#!/usr/bin/env python
"""Validate OpenAPI specification against OpenAPI 3.0 standard."""

import json
import sys
from pathlib import Path


def validate_openapi(spec_path: Path) -> bool:
    """
    Validate OpenAPI spec for common issues.

    Args:
        spec_path: Path to openapi.json file

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load OpenAPI spec: {e}")
        return False

    errors = []
    warnings = []

    # Check required fields
    if "openapi" not in spec:
        errors.append("Missing 'openapi' field")
    elif not spec["openapi"].startswith("3."):
        warnings.append(f"OpenAPI version {spec['openapi']} may not be fully supported")

    if "info" not in spec:
        errors.append("Missing 'info' field")
    else:
        info = spec["info"]
        if "title" not in info:
            errors.append("Missing 'info.title' field")
        if "version" not in info:
            errors.append("Missing 'info.version' field")
        if "description" not in info:
            warnings.append("Missing 'info.description' field (recommended)")

    if "paths" not in spec:
        errors.append("Missing 'paths' field")
    elif not spec["paths"]:
        warnings.append("No paths defined in specification")

    # Check voice refinement endpoints
    voice_endpoints = [
        "/api/projects/{project_id}/transcription-segments",
        "/api/projects/{project_id}/voice-instructions/analyze",
        "/api/projects/{project_id}/voice-instructions/regenerate",
        "/api/projects/{project_id}/voice-previews/generate",
        "/api/projects/{project_id}/voice-settings",
    ]

    missing_endpoints = []
    for endpoint in voice_endpoints:
        if endpoint not in spec.get("paths", {}):
            missing_endpoints.append(endpoint)

    if missing_endpoints:
        warnings.append(
            f"Missing {len(missing_endpoints)} voice refinement endpoints: "
            f"{', '.join(missing_endpoints[:3])}{'...' if len(missing_endpoints) > 3 else ''}"
        )

    # Check for examples in schemas
    schemas = spec.get("components", {}).get("schemas", {})
    schemas_without_examples = []
    for schema_name, schema_def in schemas.items():
        if "example" not in schema_def and "examples" not in schema_def:
            properties = schema_def.get("properties", {})
            has_field_examples = any(
                "example" in prop or "examples" in prop
                for prop in properties.values()
            )
            if not has_field_examples:
                schemas_without_examples.append(schema_name)

    if schemas_without_examples and len(schemas_without_examples) > 5:
        warnings.append(
            f"{len(schemas_without_examples)} schemas missing examples (recommended for better docs)"
        )

    # Print results
    print("\n=== OpenAPI Validation Results ===\n")
    print(f"Specification: {spec_path}")
    print(f"OpenAPI Version: {spec.get('openapi', 'unknown')}")
    print(f"API Title: {spec.get('info', {}).get('title', 'unknown')}")
    print(f"API Version: {spec.get('info', {}).get('version', 'unknown')}")
    print(f"Total Endpoints: {len(spec.get('paths', {}))}")
    print(f"Total Schemas: {len(schemas)}")

    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for error in errors:
            print(f"   - {error}")

    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"   - {warning}")

    if not errors and not warnings:
        print("\n✓ OpenAPI specification is valid!")
        return True
    elif not errors:
        print("\n✓ OpenAPI specification is valid (with warnings)")
        return True
    else:
        print("\n❌ OpenAPI specification has errors")
        return False


if __name__ == "__main__":
    spec_path = Path(__file__).parent.parent / "openapi.json"

    if not spec_path.exists():
        print(f"❌ OpenAPI spec not found at {spec_path}")
        print("Run: poetry run python scripts/export_openapi.py")
        sys.exit(1)

    is_valid = validate_openapi(spec_path)
    sys.exit(0 if is_valid else 1)
