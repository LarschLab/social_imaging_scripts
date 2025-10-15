#!/usr/bin/env python3
"""
Test script to verify manual prematch integration.

This script checks that:
1. Processing logs contain manual prematch data
2. The registration function can load and use manual prematch
"""

from pathlib import Path
from social_imaging_scripts.metadata.config import load_project_config
from social_imaging_scripts.pipeline.processing_log import (
    build_processing_log_path,
    load_processing_log,
)
from social_imaging_scripts.registration.confocal_to_anatomy import (
    _load_manual_prematch_from_log,
    _manual_prematch_to_result,
)


def test_manual_prematch_loading():
    """Test that manual prematch can be loaded from processing logs."""
    
    cfg = load_project_config()
    output_base_dir = Path(cfg.output_base_dir)
    
    # Test animals with known manual prematch data
    test_cases = [
        ("L331_f01", "L331_f01_confocal_round1_2025-05-20"),
        ("L331_f01", "L331_f01_confocal_roundn_2025-05-20"),
        ("L395_f06", "L395_f06_confocal_round1_2025-07-03"),
        ("L395_f06", "L395_f06_confocal_roundn_2025-07-03"),
    ]
    
    print("=" * 80)
    print("Testing Manual Prematch Integration")
    print("=" * 80)
    print()
    
    for animal_id, session_id in test_cases:
        print(f"Testing {animal_id}/{session_id}:")
        
        # Test 1: Check processing log exists
        log_path = build_processing_log_path(
            cfg.processing_log,
            animal_id,
            base_dir=output_base_dir,
        )
        
        if not log_path.exists():
            print(f"  ❌ Processing log not found: {log_path}")
            continue
        print(f"  ✓ Processing log exists: {log_path}")
        
        # Test 2: Load manual prematch data
        manual_prematch = _load_manual_prematch_from_log(
            animal_id,
            session_id,
            output_base_dir,
            cfg.processing_log,
        )
        
        if not manual_prematch:
            print(f"  ❌ No manual prematch data found")
            continue
        
        print(f"  ✓ Manual prematch loaded:")
        print(f"    - X translation: {manual_prematch['translation_x_px']:.1f} px")
        print(f"    - Y translation: {manual_prematch['translation_y_px']:.1f} px")
        print(f"    - Rotation: {manual_prematch['rotation_deg']:.1f}°")
        
        # Test 3: Convert to prematch result
        spacing = (1.0, 1.0, 1.0)  # Dummy spacing for test
        try:
            prematch_result = _manual_prematch_to_result(manual_prematch, spacing)
            print(f"  ✓ Converted to XYMIPPrematchResult:")
            print(f"    - Translation (µm): {prematch_result.translation_um}")
            print(f"    - Rotation: {prematch_result.rotation_deg:.1f}°")
            print(f"    - Applied: {prematch_result.applied}")
        except Exception as e:
            print(f"  ❌ Failed to convert to result: {e}")
            continue
        
        print()
    
    print("=" * 80)
    print("Integration test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_manual_prematch_loading()
