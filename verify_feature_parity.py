#!/usr/bin/env python3
"""
Comprehensive Feature Parity Verification Script
Tests all critical imports, core functionality, and backward compatibility
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class VerificationReport:
    def __init__(self):
        self.results = {
            'imports': [],
            'core_functionality': [],
            'structure': [],
            'backward_compatibility': [],
            'errors': []
        }
        self.start_time = datetime.now()

    def add_result(self, category, test_name, status, details=""):
        self.results[category].append({
            'test': test_name,
            'status': status,
            'details': details
        })

    def add_error(self, error_msg):
        self.results['errors'].append(error_msg)

    def generate_report(self):
        report = []
        report.append("# FEATURE PARITY VERIFICATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Duration: {(datetime.now() - self.start_time).total_seconds():.2f}s")
        report.append("")

        # Summary
        total_tests = sum(len(v) for k, v in self.results.items() if k != 'errors')
        passed_tests = sum(
            len([r for r in v if r['status'] == 'PASS'])
            for k, v in self.results.items() if k != 'errors'
        )
        report.append(f"## Summary")
        report.append(f"- Total Tests: {total_tests}")
        report.append(f"- Passed: {passed_tests}")
        report.append(f"- Failed: {total_tests - passed_tests}")
        report.append(f"- Errors: {len(self.results['errors'])}")
        report.append("")

        # Detailed results by category
        categories = {
            'imports': 'Import Verification',
            'core_functionality': 'Core Functionality Check',
            'structure': 'Module Structure Verification',
            'backward_compatibility': 'Backward Compatibility'
        }

        for cat_key, cat_name in categories.items():
            report.append(f"## {cat_name}")
            report.append("")
            if self.results[cat_key]:
                for result in self.results[cat_key]:
                    status_icon = "✅" if result['status'] == 'PASS' else "❌"
                    report.append(f"{status_icon} **{result['test']}**: {result['status']}")
                    if result['details']:
                        report.append(f"   - {result['details']}")
                    report.append("")
            else:
                report.append("No tests in this category.")
                report.append("")

        # Errors section
        if self.results['errors']:
            report.append("## Critical Errors")
            report.append("")
            for error in self.results['errors']:
                report.append(f"- {error}")
                report.append("")

        return "\n".join(report)

# Initialize report
report = VerificationReport()

print("=" * 80)
print("FEATURE PARITY VERIFICATION")
print("=" * 80)
print()

# ============================================================================
# 1. IMPORT VERIFICATION
# ============================================================================
print("1. IMPORT VERIFICATION")
print("-" * 80)

import_tests = [
    ('backend.core', ['BaseTool', 'ToolResult']),
    ('backend.services.file_handler', ['FileHandlerRegistry']),
    ('backend.config.prompts', ['PromptRegistry']),
    ('backend.tools.python_coder', ['python_coder_tool']),
    ('backend.tools.web_search', ['web_search_tool']),
    ('backend.tools.file_analyzer', ['file_analyzer']),
    ('backend.tools.rag_retriever', ['rag_retriever_tool']),
    ('backend.tasks.react', ['ReActAgentFactory']),
]

for module_name, items in import_tests:
    for item in items:
        test_name = f"import {item} from {module_name}"
        try:
            module = __import__(module_name, fromlist=[item])
            obj = getattr(module, item)
            print(f"✅ {test_name}")
            report.add_result('imports', test_name, 'PASS', f"Type: {type(obj).__name__}")
        except Exception as e:
            print(f"❌ {test_name}")
            print(f"   Error: {str(e)}")
            report.add_result('imports', test_name, 'FAIL', str(e))
            report.add_error(f"Import failed: {test_name} - {str(e)}")

print()

# ============================================================================
# 2. CORE FUNCTIONALITY CHECK
# ============================================================================
print("2. CORE FUNCTIONALITY CHECK")
print("-" * 80)

# Test BaseTool interface
try:
    from backend.core import BaseTool, ToolResult

    # Check BaseTool has required methods
    has_execute = hasattr(BaseTool, 'execute')
    has_name = hasattr(BaseTool, 'name')

    if has_execute and has_name:
        print("✅ BaseTool interface has required methods")
        report.add_result('core_functionality', 'BaseTool interface', 'PASS',
                         'Has execute() and name property')
    else:
        print("❌ BaseTool interface missing methods")
        report.add_result('core_functionality', 'BaseTool interface', 'FAIL',
                         f'execute: {has_execute}, name: {has_name}')
except Exception as e:
    print(f"❌ BaseTool interface check failed: {e}")
    report.add_result('core_functionality', 'BaseTool interface', 'FAIL', str(e))

# Test ToolResult creation
try:
    from backend.core import ToolResult

    result = ToolResult(
        success=True,
        output="test output",
        error=None,
        metadata={"test": "value"}
    )

    if result.success and result.output == "test output":
        print("✅ ToolResult creation and access works")
        report.add_result('core_functionality', 'ToolResult creation', 'PASS',
                         f'Created with success={result.success}')
    else:
        print("❌ ToolResult attributes incorrect")
        report.add_result('core_functionality', 'ToolResult creation', 'FAIL',
                         'Attributes not matching expected values')
except Exception as e:
    print(f"❌ ToolResult creation failed: {e}")
    report.add_result('core_functionality', 'ToolResult creation', 'FAIL', str(e))

# Test FileHandlerRegistry
try:
    from backend.services.file_handler import FileHandlerRegistry

    handlers_tested = []
    for ext in ['.csv', '.json', '.xlsx']:
        handler = FileHandlerRegistry.get_handler(ext)
        if handler:
            handlers_tested.append(ext)

    if len(handlers_tested) == 3:
        print(f"✅ FileHandlerRegistry works for: {', '.join(handlers_tested)}")
        report.add_result('core_functionality', 'FileHandlerRegistry', 'PASS',
                         f'Handlers found: {", ".join(handlers_tested)}')
    else:
        print(f"❌ FileHandlerRegistry incomplete: {handlers_tested}")
        report.add_result('core_functionality', 'FileHandlerRegistry', 'FAIL',
                         f'Only found: {", ".join(handlers_tested)}')
except Exception as e:
    print(f"❌ FileHandlerRegistry failed: {e}")
    report.add_result('core_functionality', 'FileHandlerRegistry', 'FAIL', str(e))

# Test PromptRegistry
try:
    from backend.config.prompts import PromptRegistry

    # Check available prompts
    available_prompts = PromptRegistry.list_prompts()

    # Try to get at least one prompt to verify functionality
    prompts_found = []
    if available_prompts:
        prompts_found = available_prompts[:3]  # Just check that prompts are registered

    if len(prompts_found) >= 2:  # At least 2 out of 3
        print(f"✅ PromptRegistry works, found: {len(prompts_found)}/3 prompts")
        report.add_result('core_functionality', 'PromptRegistry', 'PASS',
                         f'Found: {", ".join(prompts_found)}')
    else:
        print(f"❌ PromptRegistry incomplete: {prompts_found}")
        report.add_result('core_functionality', 'PromptRegistry', 'FAIL',
                         f'Only found: {", ".join(prompts_found)}')
except Exception as e:
    print(f"❌ PromptRegistry failed: {e}")
    report.add_result('core_functionality', 'PromptRegistry', 'FAIL', str(e))

# Test tool instances
try:
    from backend.tools.python_coder import python_coder_tool
    from backend.tools.web_search import web_search_tool
    from backend.tools.file_analyzer import file_analyzer
    from backend.tools.rag_retriever import rag_retriever_tool

    tools = {
        'python_coder_tool': python_coder_tool,
        'web_search_tool': web_search_tool,
        'file_analyzer': file_analyzer,
        'rag_retriever_tool': rag_retriever_tool
    }

    working_tools = []
    for tool_name, tool in tools.items():
        if tool is not None and hasattr(tool, 'execute'):
            working_tools.append(tool_name)

    if len(working_tools) == 4:
        print(f"✅ All tool instances functional: {', '.join(working_tools)}")
        report.add_result('core_functionality', 'Tool instances', 'PASS',
                         f'All 4 tools working')
    else:
        print(f"⚠️  Some tools working: {working_tools}")
        report.add_result('core_functionality', 'Tool instances', 'PARTIAL',
                         f'Working: {", ".join(working_tools)}')
except Exception as e:
    print(f"❌ Tool instances failed: {e}")
    report.add_result('core_functionality', 'Tool instances', 'FAIL', str(e))

# Test ReActAgentFactory
try:
    from backend.tasks.react import ReActAgentFactory

    agent = ReActAgentFactory.create_agent()

    if agent and hasattr(agent, 'execute'):
        print("✅ ReActAgentFactory creates functional agent")
        report.add_result('core_functionality', 'ReActAgentFactory', 'PASS',
                         f'Agent type: {type(agent).__name__}')
    else:
        print("❌ ReActAgentFactory agent missing execute method")
        report.add_result('core_functionality', 'ReActAgentFactory', 'FAIL',
                         'Agent missing execute method')
except Exception as e:
    print(f"❌ ReActAgentFactory failed: {e}")
    report.add_result('core_functionality', 'ReActAgentFactory', 'FAIL', str(e))

print()

# ============================================================================
# 3. MODULE STRUCTURE VERIFICATION
# ============================================================================
print("3. MODULE STRUCTURE VERIFICATION")
print("-" * 80)

directories_to_check = [
    'backend/core',
    'backend/services/file_handler',
    'backend/api/dependencies',
    'backend/api/middleware',
    'backend/tools/python_coder/executor',
    'backend/config/prompts/python_coder',
]

for dir_path in directories_to_check:
    full_path = project_root / dir_path
    if full_path.exists() and full_path.is_dir():
        print(f"✅ {dir_path}/")
        report.add_result('structure', f'Directory: {dir_path}/', 'PASS',
                         f'Exists at {full_path}')
    else:
        print(f"❌ {dir_path}/ - NOT FOUND")
        report.add_result('structure', f'Directory: {dir_path}/', 'FAIL',
                         'Directory not found')

# Check for key files
key_files = [
    'backend/core/__init__.py',
    'backend/core/base_tool.py',
    'backend/services/file_handler/__init__.py',
    'backend/services/file_handler/registry.py',
    'backend/config/prompts/__init__.py',
    'backend/config/prompts/registry.py',
]

for file_path in key_files:
    full_path = project_root / file_path
    if full_path.exists() and full_path.is_file():
        print(f"✅ {file_path}")
        report.add_result('structure', f'File: {file_path}', 'PASS',
                         f'Size: {full_path.stat().st_size} bytes')
    else:
        print(f"❌ {file_path} - NOT FOUND")
        report.add_result('structure', f'File: {file_path}', 'FAIL',
                         'File not found')

# Check backward compatibility shims exist
compat_shims = [
    ('backend/tasks/React.py', 'Backward compat shim: React.py'),
    ('backend/tools/python_coder_tool.py', 'Backward compat shim: python_coder_tool.py'),
    ('backend/tools/file_analyzer_tool.py', 'Backward compat shim: file_analyzer_tool.py'),
]

for file_path, description in compat_shims:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"✅ {description}")
        report.add_result('structure', description, 'PASS',
                         f'Found at {file_path}')
    else:
        print(f"❌ {description} - NOT FOUND")
        report.add_result('structure', description, 'FAIL',
                         'Backward compatibility shim missing')

print()

# ============================================================================
# 4. BACKWARD COMPATIBILITY
# ============================================================================
print("4. BACKWARD COMPATIBILITY")
print("-" * 80)

# Test old imports with deprecation warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Test old ReAct import
    try:
        from backend.tasks.React import ReActAgent
        print("✅ Old import: from backend.tasks.React import ReActAgent")

        # Check if deprecation warning was raised
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        if deprecation_warnings:
            print(f"   ⚠️  Deprecation warning raised: {deprecation_warnings[0].message}")
            report.add_result('backward_compatibility', 'Old ReActAgent import', 'PASS',
                             'Works with deprecation warning')
        else:
            print("   ℹ️  No deprecation warning (may be intentional)")
            report.add_result('backward_compatibility', 'Old ReActAgent import', 'PASS',
                             'Works without deprecation warning')
    except Exception as e:
        print(f"❌ Old ReActAgent import failed: {e}")
        report.add_result('backward_compatibility', 'Old ReActAgent import', 'FAIL', str(e))

# Test legacy singleton
try:
    from backend.tasks.react import react_agent

    if react_agent and hasattr(react_agent, 'execute'):
        print("✅ Legacy singleton 'react_agent' still works")
        report.add_result('backward_compatibility', 'Legacy react_agent singleton', 'PASS',
                         f'Type: {type(react_agent).__name__}')
    else:
        print("❌ Legacy singleton 'react_agent' not functional")
        report.add_result('backward_compatibility', 'Legacy react_agent singleton', 'FAIL',
                         'Missing execute method')
except Exception as e:
    print(f"❌ Legacy singleton failed: {e}")
    report.add_result('backward_compatibility', 'Legacy react_agent singleton', 'FAIL', str(e))

# Test old tool imports
old_tool_imports = [
    ('backend.tools.python_coder_tool', 'python_coder_tool'),
    ('backend.tools.file_analyzer_tool', 'file_analyzer'),
]

for module_name, item_name in old_tool_imports:
    try:
        module = __import__(module_name, fromlist=[item_name])
        obj = getattr(module, item_name)
        print(f"✅ Old import: from {module_name} import {item_name}")
        report.add_result('backward_compatibility', f'Old {item_name} import', 'PASS',
                         f'Type: {type(obj).__name__}')
    except Exception as e:
        print(f"⚠️  Old import failed: from {module_name} import {item_name}")
        print(f"   {str(e)}")
        report.add_result('backward_compatibility', f'Old {item_name} import', 'WARN',
                         f'Not available: {str(e)}')

print()
print("=" * 80)

# Generate and save report
report_content = report.generate_report()
report_path = project_root / 'VERIFICATION_FEATURE_PARITY.md'

with open(report_path, 'w') as f:
    f.write(report_content)

print(f"Report saved to: {report_path}")
print()
print("VERIFICATION COMPLETE")
print("=" * 80)
