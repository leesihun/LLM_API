#!/usr/bin/env python3
"""
Integration Testing and Quality Checks Script
Performs comprehensive verification of the refactored codebase
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import ast
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config.prompts.registry import PromptRegistry
from backend.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")

def print_success(msg: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

def print_error(msg: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")

def print_warning(msg: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")

def print_info(msg: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")


# =============================================================================
# 1. PromptRegistry Quality Check
# =============================================================================

def check_prompt_registry():
    """Check PromptRegistry validation and adoption"""
    print_section("1. PromptRegistry Quality Check")

    # List all prompts
    all_prompts = PromptRegistry.list_all()
    print_info(f"Total registered prompts: {len(all_prompts)}")

    # Validate all prompts
    validation_results = PromptRegistry.validate_all()
    passed = sum(1 for v in validation_results.values() if v)
    failed = len(validation_results) - passed

    print_info(f"Validation results: {passed} passed, {failed} failed")

    # Show all prompts with details
    print("\nRegistered Prompts:")
    for prompt_name in sorted(all_prompts):
        try:
            info = PromptRegistry.get_info(prompt_name)
            params = info['parameters']
            required_params = [k for k, v in params.items() if v['required']]
            print(f"  • {prompt_name}")
            print(f"    - Module: {info['module']}")
            print(f"    - Required params: {required_params if required_params else 'None'}")
        except Exception as e:
            print_error(f"  • {prompt_name}: Error getting info - {e}")

    # Check adoption rate by searching for PromptRegistry.get() calls
    print("\nChecking adoption rate...")
    project_root = Path(__file__).parent
    py_files = list(project_root.rglob("*.py"))

    registry_usage_count = 0
    old_pattern_count = 0

    for py_file in py_files:
        if 'venv' in str(py_file) or '.git' in str(py_file):
            continue
        try:
            content = py_file.read_text()
            registry_usage_count += content.count("PromptRegistry.get(")
            # Count old patterns (hardcoded prompts)
            if 'def get_' in content and 'prompt' in content.lower():
                old_pattern_count += 1
        except:
            pass

    adoption_rate = registry_usage_count / max(1, registry_usage_count + old_pattern_count) * 100
    print_info(f"PromptRegistry.get() calls: {registry_usage_count}")
    print_info(f"Old pattern files: {old_pattern_count}")
    print_info(f"Adoption rate: {adoption_rate:.1f}%")

    if adoption_rate >= 50:
        print_success(f"Adoption rate target met: {adoption_rate:.1f}% >= 50%")
    else:
        print_warning(f"Adoption rate below target: {adoption_rate:.1f}% < 50%")

    return {
        'total_prompts': len(all_prompts),
        'validation_passed': passed,
        'validation_failed': failed,
        'adoption_rate': adoption_rate,
        'registry_calls': registry_usage_count
    }


# =============================================================================
# 2. Import Analysis
# =============================================================================

def check_imports():
    """Check all imports across codebase"""
    print_section("2. Import Analysis Across Codebase")

    project_root = Path(__file__).parent
    py_files = [f for f in project_root.rglob("*.py")
                if 'venv' not in str(f) and '.git' not in str(f)]

    old_imports = {
        'backend.tasks.React': [],
        'backend.tools.python_coder_tool': [],
        'backend.tools.file_analyzer_tool': [],
        'backend.tools.web_search.tool': [],
        'backend.api.routes': [],
        'backend.core.agent_graph': []
    }

    new_imports = {
        'backend.tasks.react': [],
        'backend.tools.python_coder': [],
        'backend.tools.file_analyzer': [],
        'backend.tools.web_search': [],
        'backend.api.routes': [],
        'backend.config.prompts.registry': []
    }

    for py_file in py_files:
        try:
            content = py_file.read_text()

            # Check for old imports
            for old_pattern in old_imports.keys():
                if f'from {old_pattern}' in content or f'import {old_pattern}' in content:
                    old_imports[old_pattern].append(str(py_file))

            # Check for new imports
            for new_pattern in new_imports.keys():
                if f'from {new_pattern}' in content or f'import {new_pattern}' in content:
                    new_imports[new_pattern].append(str(py_file))
        except:
            pass

    print("\nOld Import Patterns Found:")
    total_old = 0
    for pattern, files in old_imports.items():
        if files:
            print_warning(f"{pattern}: {len(files)} files")
            total_old += len(files)
            for f in files[:3]:  # Show first 3
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        else:
            print_success(f"{pattern}: No old imports found")

    print("\nNew Import Patterns Found:")
    total_new = 0
    for pattern, files in new_imports.items():
        if files:
            print_success(f"{pattern}: {len(files)} files")
            total_new += len(files)
        else:
            print_info(f"{pattern}: Not used yet")

    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print_info(f"Old import usage: {total_old} occurrences")
    print_info(f"New import usage: {total_new} occurrences")

    if total_old == 0:
        print_success("All imports migrated to new locations!")
    else:
        print_warning(f"{total_old} old imports still exist")

    return {
        'old_imports_count': total_old,
        'new_imports_count': total_new,
        'old_import_details': old_imports,
        'new_import_details': new_imports
    }


# =============================================================================
# 3. Security Check
# =============================================================================

def check_security():
    """Check security implementations"""
    print_section("3. Security Implementation Check")

    project_root = Path(__file__).parent

    checks = {
        'password_hashing': False,
        'security_headers': False,
        'rbac_dependencies': False,
        'file_validation': False
    }

    # Check password hashing in auth.py
    auth_file = project_root / "backend/utils/auth.py"
    if auth_file.exists():
        content = auth_file.read_text()
        if 'bcrypt' in content or 'hash_password' in content:
            checks['password_hashing'] = True
            print_success("Password hashing implemented in backend/utils/auth.py")
        else:
            print_error("Password hashing NOT found in backend/utils/auth.py")
    else:
        print_error("backend/utils/auth.py not found")

    # Check security headers middleware
    middleware_file = project_root / "backend/api/middleware/security_headers.py"
    if middleware_file.exists():
        content = middleware_file.read_text()
        if 'X-Content-Type-Options' in content or 'SecurityHeadersMiddleware' in content:
            checks['security_headers'] = True
            print_success("Security headers middleware found")
        else:
            print_warning("Security headers file exists but may be incomplete")
    else:
        print_error("backend/api/middleware/security_headers.py not found")

    # Check RBAC dependencies
    role_checker = project_root / "backend/api/dependencies/role_checker.py"
    if role_checker.exists():
        checks['rbac_dependencies'] = True
        print_success("RBAC role checker dependency exists")
    else:
        print_error("backend/api/dependencies/role_checker.py not found")

    # Check file validation utilities
    validators = [
        project_root / "backend/utils/validators.py",
        project_root / "backend/config/prompts/validators.py",
        project_root / "backend/services/file_handler/utils.py"
    ]

    for validator in validators:
        if validator.exists():
            checks['file_validation'] = True
            print_success(f"File validation found: {validator.name}")
            break
    else:
        print_error("No file validation utilities found")

    passed = sum(checks.values())
    total = len(checks)

    print(f"\n{Colors.BOLD}Security Score: {passed}/{total}{Colors.RESET}")

    if passed == total:
        print_success("All security checks passed!")
    else:
        print_warning(f"{total - passed} security checks failed")

    return checks


# =============================================================================
# 4. Architecture Layering Check
# =============================================================================

def check_architecture_layers():
    """Check for circular dependencies and layer violations"""
    print_section("4. Architecture Layering & Dependencies")

    project_root = Path(__file__).parent

    # Define layers
    layers = {
        'core': ['backend/core'],
        'models': ['backend/models'],
        'config': ['backend/config'],
        'utils': ['backend/utils'],
        'services': ['backend/services'],
        'tools': ['backend/tools'],
        'tasks': ['backend/tasks'],
        'api': ['backend/api']
    }

    # Layer rules: key cannot import from values
    layer_rules = {
        'core': ['services', 'tools', 'tasks', 'api'],
        'models': ['services', 'tools', 'tasks', 'api'],
        'config': ['services', 'tools', 'tasks', 'api'],
        'utils': ['services', 'tools', 'tasks', 'api'],
        'services': ['tasks', 'api'],
        'tools': ['tasks', 'api'],
        'tasks': ['api']
    }

    violations = []

    for layer_name, layer_paths in layers.items():
        if layer_name not in layer_rules:
            continue

        forbidden_imports = layer_rules[layer_name]

        for layer_path in layer_paths:
            full_path = project_root / layer_path
            if not full_path.exists():
                continue

            py_files = list(full_path.rglob("*.py"))

            for py_file in py_files:
                try:
                    content = py_file.read_text()

                    for forbidden_layer in forbidden_imports:
                        for forbidden_path in layers.get(forbidden_layer, []):
                            forbidden_module = forbidden_path.replace('/', '.')

                            if f'from {forbidden_module}' in content or f'import {forbidden_module}' in content:
                                violations.append({
                                    'file': str(py_file.relative_to(project_root)),
                                    'layer': layer_name,
                                    'imports_from': forbidden_layer,
                                    'rule': f"{layer_name} should not import from {forbidden_layer}"
                                })
                except:
                    pass

    print(f"Checking layer violations...")
    print_info(f"Total violations found: {len(violations)}")

    if violations:
        print_warning("\nLayer Violations:")
        for v in violations[:10]:  # Show first 10
            print(f"  • {v['file']}")
            print(f"    Rule: {v['rule']}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations")
    else:
        print_success("No layer violations found! Architecture is clean.")

    return {
        'violations_count': len(violations),
        'violations': violations
    }


# =============================================================================
# 5. File Count Analysis
# =============================================================================

def check_file_counts():
    """Count files in each directory"""
    print_section("5. File Count Analysis")

    project_root = Path(__file__).parent

    directories = [
        'backend/core',
        'backend/services',
        'backend/tools',
        'backend/tasks',
        'backend/api',
        'backend/config/prompts',
        'backend/utils',
        'backend/models'
    ]

    counts = {}

    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            py_files = list(dir_path.rglob("*.py"))
            counts[directory] = len(py_files)
            print_info(f"{directory}: {len(py_files)} Python files")
        else:
            counts[directory] = 0
            print_warning(f"{directory}: Directory not found")

    total = sum(counts.values())
    print(f"\n{Colors.BOLD}Total Python files in tracked directories: {total}{Colors.RESET}")

    return counts


# =============================================================================
# 6. Dead Code Check
# =============================================================================

def check_dead_code():
    """Check for dead code and orphaned imports"""
    print_section("6. Dead Code & Orphaned Imports Check")

    project_root = Path(__file__).parent

    # Check if agent_graph moved to legacy
    agent_graph_legacy = project_root / "backend/tasks/legacy/agent_graph.py"
    agent_graph_old = project_root / "backend/core/agent_graph.py"

    print("\nChecking agent_graph.py location:")
    if agent_graph_legacy.exists():
        print_success(f"agent_graph.py found in legacy: {agent_graph_legacy}")
    else:
        print_warning("agent_graph.py NOT in backend/tasks/legacy/")

    if agent_graph_old.exists():
        print_warning(f"agent_graph.py still exists in old location: {agent_graph_old}")
    else:
        print_success("agent_graph.py removed from backend/core/")

    # Check for references to removed code
    print("\nChecking for references to removed/legacy code:")

    removed_patterns = [
        'from backend.core.agent_graph',
        'import backend.core.agent_graph',
        'from backend.tasks.React import',  # Old monolithic
        'from backend.tools.python_coder_tool import',  # Old monolithic
        'python_executor_engine',  # Removed file
        'PythonExecutorEngine'  # Removed class
    ]

    py_files = [f for f in project_root.rglob("*.py")
                if 'venv' not in str(f) and '.git' not in str(f) and 'legacy' not in str(f)]

    references_found = {pattern: [] for pattern in removed_patterns}

    for py_file in py_files:
        try:
            content = py_file.read_text()
            for pattern in removed_patterns:
                if pattern in content:
                    references_found[pattern].append(str(py_file.relative_to(project_root)))
        except:
            pass

    total_orphaned = 0
    for pattern, files in references_found.items():
        if files:
            print_error(f"References to '{pattern}': {len(files)} files")
            total_orphaned += len(files)
            for f in files[:3]:
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        else:
            print_success(f"No references to '{pattern}'")

    if total_orphaned == 0:
        print_success("\nNo orphaned imports found!")
    else:
        print_warning(f"\n{total_orphaned} orphaned imports need cleanup")

    return {
        'agent_graph_in_legacy': agent_graph_legacy.exists(),
        'agent_graph_in_old_location': agent_graph_old.exists(),
        'orphaned_imports': total_orphaned,
        'orphaned_details': references_found
    }


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all integration tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                  INTEGRATION TESTING & QUALITY CHECKS                        ║")
    print("║                         LLM_API Refactoring v2.0                             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")

    results = {}

    try:
        results['prompt_registry'] = check_prompt_registry()
    except Exception as e:
        print_error(f"PromptRegistry check failed: {e}")
        results['prompt_registry'] = {'error': str(e)}

    try:
        results['imports'] = check_imports()
    except Exception as e:
        print_error(f"Import check failed: {e}")
        results['imports'] = {'error': str(e)}

    try:
        results['security'] = check_security()
    except Exception as e:
        print_error(f"Security check failed: {e}")
        results['security'] = {'error': str(e)}

    try:
        results['architecture'] = check_architecture_layers()
    except Exception as e:
        print_error(f"Architecture check failed: {e}")
        results['architecture'] = {'error': str(e)}

    try:
        results['file_counts'] = check_file_counts()
    except Exception as e:
        print_error(f"File count check failed: {e}")
        results['file_counts'] = {'error': str(e)}

    try:
        results['dead_code'] = check_dead_code()
    except Exception as e:
        print_error(f"Dead code check failed: {e}")
        results['dead_code'] = {'error': str(e)}

    # Final Summary
    print_section("FINAL SUMMARY")

    total_checks = 0
    passed_checks = 0

    # PromptRegistry
    if 'prompt_registry' in results and 'error' not in results['prompt_registry']:
        total_checks += 1
        if results['prompt_registry'].get('adoption_rate', 0) >= 50:
            passed_checks += 1
            print_success("PromptRegistry: Adoption rate >= 50%")
        else:
            print_warning(f"PromptRegistry: Adoption rate {results['prompt_registry'].get('adoption_rate', 0):.1f}%")

    # Imports
    if 'imports' in results and 'error' not in results['imports']:
        total_checks += 1
        if results['imports'].get('old_imports_count', 1) == 0:
            passed_checks += 1
            print_success("Imports: All migrated to new locations")
        else:
            print_warning(f"Imports: {results['imports'].get('old_imports_count', 0)} old imports remaining")

    # Security
    if 'security' in results and 'error' not in results['security']:
        security_passed = sum(results['security'].values())
        security_total = len(results['security'])
        total_checks += 1
        if security_passed == security_total:
            passed_checks += 1
            print_success(f"Security: All {security_total} checks passed")
        else:
            print_warning(f"Security: {security_passed}/{security_total} checks passed")

    # Architecture
    if 'architecture' in results and 'error' not in results['architecture']:
        total_checks += 1
        if results['architecture'].get('violations_count', 1) == 0:
            passed_checks += 1
            print_success("Architecture: No layer violations")
        else:
            print_warning(f"Architecture: {results['architecture'].get('violations_count', 0)} violations")

    # Dead Code
    if 'dead_code' in results and 'error' not in results['dead_code']:
        total_checks += 1
        if results['dead_code'].get('orphaned_imports', 1) == 0:
            passed_checks += 1
            print_success("Dead Code: No orphaned imports")
        else:
            print_warning(f"Dead Code: {results['dead_code'].get('orphaned_imports', 0)} orphaned imports")

    print(f"\n{Colors.BOLD}Overall Score: {passed_checks}/{total_checks} checks passed{Colors.RESET}")

    if passed_checks == total_checks:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL CHECKS PASSED! Codebase is in excellent shape.{Colors.RESET}\n")
    elif passed_checks >= total_checks * 0.8:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️ MOSTLY PASSED. Some minor issues to address.{Colors.RESET}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ NEEDS WORK. Several issues require attention.{Colors.RESET}\n")

    # Save results to JSON
    output_file = Path(__file__).parent / "integration_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print_info(f"Detailed results saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = run_all_tests()
