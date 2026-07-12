#!/usr/bin/env python
"""Export OpenAPI specification to JSON and YAML files."""

import json
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. YAML export will be skipped.")
    print("Install with: poetry add pyyaml --group dev")

from app.main import app


def export_openapi():
    """Export OpenAPI spec to openapi.json and openapi.yaml."""
    # Get OpenAPI schema
    schema = app.openapi()

    # Create output directory
    output_dir = Path(__file__).parent.parent

    # Export JSON
    json_path = output_dir / "openapi.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"✓ Exported OpenAPI spec to {json_path}")

    # Export YAML (if available)
    if YAML_AVAILABLE:
        yaml_path = output_dir / "openapi.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(schema, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"✓ Exported OpenAPI spec to {yaml_path}")

    # Print summary
    print("\nOpenAPI Specification Summary:")
    print(f"  Title: {schema['info']['title']}")
    print(f"  Version: {schema['info']['version']}")
    print(f"  Description: {schema['info']['description'][:80]}...")
    print(f"  Total endpoints: {len(schema['paths'])}")
    print(f"  Tags: {', '.join(tag['name'] for tag in schema.get('tags', []))}")

    # Print voice-refinement endpoints
    voice_refinement_paths = [
        path for path in schema['paths'].keys()
        if 'voice' in path or 'transcription-segments' in path
    ]
    if voice_refinement_paths:
        print(f"\nVoice Refinement Endpoints ({len(voice_refinement_paths)}):")
        for path in sorted(voice_refinement_paths):
            methods = list(schema['paths'][path].keys())
            print(f"  {', '.join(m.upper() for m in methods if m != 'parameters')}: {path}")


if __name__ == "__main__":
    export_openapi()
