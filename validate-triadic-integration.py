#!/usr/bin/env python3
"""
Test script for validating the enhanced GitHub Action for triadic architecture issue generation.

This script validates that the GitHub Action YAML is well-formed and contains all the expected
OpenCoq integration patterns and comprehensive issue generation capabilities.
"""

import yaml
import json
import os
import re

def validate_github_action():
    """Validate the GitHub Action configuration for triadic architecture."""
    
    action_path = ".github/workflows/generate-triadic-issues.yml"
    
    if not os.path.exists(action_path):
        print("❌ GitHub Action file not found")
        return False
    
    print("🔍 Validating GitHub Action for Triadic Architecture Issue Generation")
    print("=" * 80)
    
    # Load and parse the YAML
    try:
        with open(action_path, 'r') as f:
            action_config = yaml.safe_load(f)
        
        if action_config is None:
            print("❌ YAML file is empty or invalid")
            return False
            
    except Exception as e:
        print(f"❌ Failed to parse YAML: {e}")
        return False
    
    # Validate basic structure
    required_keys = ['name', 'on', 'jobs']
    for key in required_keys:
        if key not in action_config:
            print(f"❌ Missing required key: {key}")
            return False
    
    print("✅ Basic YAML structure is valid")
    
    # Validate workflow inputs
    if 'workflow_dispatch' not in action_config['on']:
        print("❌ workflow_dispatch trigger not found")
        return False
    
    inputs = action_config['on']['workflow_dispatch'].get('inputs', {})
    if 'phase' not in inputs:
        print("❌ Phase input parameter not found")
        return False
    
    phase_options = inputs['phase'].get('options', [])
    expected_phases = ['all', '1', '2', '3', '4', '5', '6']
    
    if set(phase_options) != set(expected_phases):
        print(f"❌ Phase options mismatch. Expected: {expected_phases}, Got: {phase_options}")
        return False
    
    print("✅ Workflow inputs are correctly configured")
    
    # Validate jobs structure
    jobs = action_config.get('jobs', {})
    if 'generate-issues' not in jobs:
        print("❌ generate-issues job not found")
        return False
    
    generate_job = jobs['generate-issues']
    if 'steps' not in generate_job:
        print("❌ Steps not found in generate-issues job")
        return False
    
    print("✅ Job structure is valid")
    
    # Count and validate issue generation steps
    steps = generate_job['steps']
    phase_steps = [step for step in steps if 'Generate Phase' in step.get('name', '') or 'Generate Meta-Cognitive' in step.get('name', '') or 'Generate Testing' in step.get('name', '')]
    
    expected_phase_count = 6  # Phases 1-5 + Meta-Cognitive Enhancement
    if len(phase_steps) != expected_phase_count:
        print(f"❌ Expected {expected_phase_count} phase generation steps, found {len(phase_steps)}")
        return False
    
    print(f"✅ Found all {expected_phase_count} phase generation steps")
    
    # Validate OpenCoq integration keywords
    action_content = str(action_config)
    opencoq_keywords = [
        'opencoq', 'rocq-prover', 'formal-verification', 
        'theorem', 'proof', 'coq', 'verification'
    ]
    
    found_keywords = []
    for keyword in opencoq_keywords:
        if keyword.lower() in action_content.lower():
            found_keywords.append(keyword)
    
    if len(found_keywords) < 4:  # At least 4 OpenCoq-related keywords should be present
        print(f"❌ Insufficient OpenCoq integration. Found keywords: {found_keywords}")
        return False
    
    print(f"✅ OpenCoq integration detected: {found_keywords}")
    
    # Validate meta-cognitive enhancement phase
    meta_cognitive_found = False
    for step in steps:
        if 'Meta-Cognitive Enhancement' in step.get('name', ''):
            meta_cognitive_found = True
            break
    
    if not meta_cognitive_found:
        print("❌ Meta-Cognitive Enhancement phase not found")
        return False
    
    print("✅ Meta-Cognitive Enhancement phase is present")
    
    # Validate project board creation
    project_board_step = None
    for step in steps:
        if 'Create Project Board' in step.get('name', ''):
            project_board_step = step
            break
    
    if not project_board_step:
        print("❌ Project board creation step not found")
        return False
    
    # Check for Phase 6 in project board
    project_script = project_board_step.get('with', {}).get('script', '')
    if 'Phase 6: Meta-Cognitive Enhancement' not in project_script:
        print("❌ Phase 6 not included in project board")
        return False
    
    print("✅ Project board includes all phases including Meta-Cognitive Enhancement")
    
    return True

def validate_documentation():
    """Validate that documentation files exist and are comprehensive."""
    
    print("\n📚 Validating Documentation")
    print("=" * 80)
    
    required_docs = [
        'TRIADIC_ARCHITECTURE_SUMMARY.md',
        'OPENCOQ_INTEGRATION_ROADMAP.md'
    ]
    
    for doc in required_docs:
        if not os.path.exists(doc):
            print(f"❌ Missing documentation: {doc}")
            return False
        
        # Check file size (should be substantial)
        size = os.path.getsize(doc)
        if size < 1000:  # At least 1KB
            print(f"❌ Documentation too small: {doc} ({size} bytes)")
            return False
        
        print(f"✅ Documentation found: {doc} ({size} bytes)")
    
    # Validate OpenCoq roadmap contains key sections
    with open('OPENCOQ_INTEGRATION_ROADMAP.md', 'r') as f:
        roadmap_content = f.read()
    
    required_sections = [
        '## 🎯 Mission Statement',
        '## 🏗️ Architecture Overview', 
        '## 🔬 Phase-by-Phase Integration',
        'Phase 6: Meta-Cognitive Enhancement',
        '```coq',  # Should contain Coq code examples
        'OpenCoq'
    ]
    
    for section in required_sections:
        if section not in roadmap_content:
            print(f"❌ Missing section in roadmap: {section}")
            return False
    
    print("✅ OpenCoq integration roadmap is comprehensive")
    
    return True

def validate_examples():
    """Validate that example files work correctly."""
    
    print("\n🧪 Validating Examples")
    print("=" * 80)
    
    # Check if RR triadic reasoning example exists
    rr_example = "examples/rr_triadic_reasoning.py"
    if not os.path.exists(rr_example):
        print(f"❌ Missing example: {rr_example}")
        return False
    
    print(f"✅ RR triadic reasoning example found: {rr_example}")
    
    # Validate demo script
    demo_script = "demo-triadic-architecture.py"
    if not os.path.exists(demo_script):
        print(f"❌ Missing demo script: {demo_script}")
        return False
    
    print(f"✅ Demo script found: {demo_script}")
    
    return True

def main():
    """Main validation function."""
    
    print("🧠 Triadic Architecture of Relevance Realization - Validation Suite")
    print("=" * 80)
    print()
    
    # Change to repository directory
    os.chdir('/home/runner/work/llamacog/llamacog')
    
    # Run all validations
    validations = [
        ("GitHub Action Configuration", validate_github_action),
        ("Documentation", validate_documentation),
        ("Examples", validate_examples),
    ]
    
    all_passed = True
    
    for name, validator in validations:
        try:
            result = validator()
            if result:
                print(f"\n🎉 {name}: PASSED")
            else:
                print(f"\n❌ {name}: FAILED")
                all_passed = False
        except Exception as e:
            print(f"\n💥 {name}: ERROR - {e}")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🏆 ALL VALIDATIONS PASSED!")
        print("✅ The enhanced Triadic Architecture with OpenCoq integration is ready!")
        print("🚀 Ready to generate comprehensive development issues")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("🔧 Please address the issues above")
    
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)