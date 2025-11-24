# LLM_API 전체 리팩토링 계획

## 분석 결과 요약

### 발견된 주요 문제점

#### 1. 심각한 코드 중복 (Critical)
- **파일 핸들러 60% 중복**: `python_coder/file_handlers/` vs `file_analyzer/handlers/`
- **PromptRegistry 이중 구현**: `config/prompts/__init__.py` vs `utils/prompt_builder.py`
- **파일 메타데이터 추출 3곳**: FileMetadataService, FileHandlerFactory, analyzer handlers
- **컨텍스트 포맷팅 중복**: ContextManager vs AnswerGenerator
- **에러 핸들링 패턴 163회 반복**: 28개 파일에 동일한 try-except 블록

#### 2. 비대한 파일 (Monolithic Files)
- `backend/tools/python_coder/executor.py`: **843 줄**
- `backend/config/prompts/python_coder.py`: **794 줄** (프롬프트만!)
- `backend/tools/python_coder/orchestrator.py`: **788 줄**
- `backend/api/routes/chat.py`: **509 줄**
- `backend/tasks/react/agent.py`: **501 줄**

#### 3. 아키텍처 문제
- **3개 에이전트 패턴** 존재: ReAct (활성), Plan-Execute (활성), Agent Graph (미사용)
- **SmartAgent 라우터 고장**: AUTO 모드가 항상 Plan-Execute 선택
- **전역 싱글톤**: ReAct agent가 전역 인스턴스로 요청 간 상태 공유 위험
- **도구 인터페이스 불일치**: 각 도구마다 다른 시그니처, 반환 타입, async/sync
- **순환 의존성 위험**: Config → Prompts 런타임 임포트

#### 4. 보안 취약점 (Security)
- **평문 비밀번호 저장**: `auth.py` 73, 98줄
- **파일 업로드 검증 없음**: 크기 제한, 타입 제한 미적용
- **세션 소유권 미검증**: 다른 사용자의 파일 접근 가능
- **JWT 시크릿 미회전**: 하드코딩된 키

#### 5. 프롬프트 문제
- **극단적 장황함**: Python coder 프롬프트 **328 줄** (정상 모드)
- **낮은 채택률**: PromptRegistry 사용 **16%** (14/90 파일)
- **하드코딩된 프롬프트**: `verification.py:124`, `phase_manager.py:108`
- **상충되는 지침**: Plan-Execute "finish step 포함하지 말라" vs ReAct "finish로 종료"
- **과도한 검증**: 오탐 증가로 불필요한 재시도 유발

---

## 리팩토링 아키텍처 설계

### 계층 구조 (Layered Architecture)

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: API (FastAPI Routes)                      │
│  - 얇은 컨트롤러 레이어                                 │
│  - 요청 검증, 인증, 응답 포맷팅만 담당                     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3: Services (Business Logic)                 │
│  - ChatService, FileService, AuthService            │
│  - 비즈니스 로직 집중화                                 │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Agents & Tools                            │
│  - ReAct Agent (주요 에이전트)                         │
│  - Tools: BaseTool 인터페이스 구현                     │
│  - 표준화된 ToolResult 반환                            │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 1: Core Infrastructure                       │
│  - Config, Utils, Models                            │
│  - 공유 서비스 (FileHandler, PromptRegistry, LLM)    │
│  - 상위 레이어 의존성 금지                               │
└─────────────────────────────────────────────────────┘
```

---

## 새로운 디렉토리 구조

```
backend/
├── core/                           # Layer 1: 핵심 인프라
│   ├── __init__.py
│   ├── base_tool.py               # ✨ 신규: BaseTool 인터페이스
│   ├── exceptions.py              # ✨ 신규: 커스텀 예외 계층
│   ├── result_types.py            # ✨ 신규: 표준 Result 타입
│   └── retry.py                   # ✨ 신규: 재시도 유틸리티
│
├── config/
│   ├── settings.py                # 📝 수정: 순환 의존성 제거
│   └── prompts/                   # 📝 대폭 개선
│       ├── __init__.py            # PromptRegistry (단일 버전)
│       ├── registry.py            # ✨ 신규: 레지스트리 로직
│       ├── validators.py          # ✨ 신규: 프롬프트 검증
│       ├── task_classification.py # 📝 개선: 예제 추가
│       ├── react_agent.py         # 📝 축소: 70→40줄
│       ├── python_coder/          # 📝 분할: 328줄→모듈화
│       │   ├── __init__.py
│       │   ├── generation.py      # 생성 프롬프트 (~100줄)
│       │   ├── verification.py    # 검증 프롬프트 (~80줄)
│       │   ├── fixing.py          # 수정 프롬프트 (~60줄)
│       │   └── templates.py       # 공통 템플릿 (~40줄)
│       ├── plan_execute.py        # 📝 수정: finish step 명확화
│       ├── web_search.py
│       ├── file_analyzer.py
│       ├── phase_manager.py       # ✨ 신규: 하드코딩 제거
│       └── context_formatting.py  # ✨ 신규: 컨텍스트 프롬프트
│
├── utils/
│   ├── llm_factory.py             # ✅ 유지
│   ├── logging_utils.py           # ✅ 유지
│   ├── prompt_builder.py          # ❌ 삭제: 중복 제거
│   ├── auth.py                    # 📝 수정: 비밀번호 해싱
│   ├── phase_manager.py           # 📝 수정: 프롬프트 중앙화
│   ├── error_handler.py           # ✨ 신규: 통합 에러 핸들러
│   ├── file_utils.py              # ✨ 신규: 파일 유틸리티
│   └── validators.py              # ✨ 신규: 입력 검증
│
├── models/
│   ├── schemas.py                 # 📝 정리: 중복 모델 제거
│   ├── tool_metadata.py           # 📝 정리: 단일 정의
│   └── api_responses.py           # ✨ 신규: 표준 API 응답
│
├── services/                      # ✨ 신규 레이어
│   ├── __init__.py
│   ├── chat_service.py            # 채팅 비즈니스 로직
│   ├── file_service.py            # 파일 관리 로직
│   ├── auth_service.py            # 인증 로직
│   ├── conversation_service.py    # 대화 관리
│   └── file_handler/              # 🔄 통합 파일 핸들러
│       ├── __init__.py
│       ├── base.py                # 통합 BaseFileHandler
│       ├── registry.py            # 핸들러 레지스트리
│       ├── csv_handler.py         # 단일 CSV 핸들러
│       ├── excel_handler.py       # 단일 Excel 핸들러
│       ├── json_handler.py        # 단일 JSON 핸들러
│       ├── text_handler.py        # 단일 Text 핸들러
│       ├── pdf_handler.py         # 단일 PDF 핸들러
│       ├── image_handler.py       # 단일 Image 핸들러
│       └── utils.py               # 공통 유틸 (파일 크기 등)
│
├── tools/
│   ├── __init__.py
│   ├── python_coder/
│   │   ├── __init__.py
│   │   ├── tool.py                # 📝 수정: BaseTool 상속
│   │   ├── orchestrator.py        # 📝 분할: 788→400줄
│   │   ├── file_preparation.py    # ✨ 신규: 파일 준비 로직
│   │   ├── execution_pipeline.py  # ✨ 신규: 실행 파이프라인
│   │   ├── generator.py           # 📝 수정: FileService 사용
│   │   ├── executor/              # 📝 분할: 843줄→모듈화
│   │   │   ├── __init__.py
│   │   │   ├── core.py            # 핵심 실행 (~200줄)
│   │   │   ├── import_validator.py # 임포트 검증 (~150줄)
│   │   │   ├── repl_manager.py    # REPL 관리 (~150줄)
│   │   │   ├── sandbox.py         # 샌드박스 설정 (~100줄)
│   │   │   └── utils.py           # 유틸리티 (~100줄)
│   │   ├── verifier.py            # 📝 수정: 검증 완화
│   │   ├── fixer.py               # ✅ 유지
│   │   └── models.py              # ✅ 유지
│   ├── file_analyzer/
│   │   ├── __init__.py
│   │   ├── tool.py                # 📝 수정: BaseTool 상속
│   │   ├── analyzer.py            # 📝 수정: FileService 사용
│   │   ├── llm_analyzer.py        # ✅ 유지
│   │   └── models.py              # ✅ 유지
│   │   # handlers/ 폴더 삭제 → services/file_handler 사용
│   ├── web_search/
│   │   ├── __init__.py
│   │   ├── tool.py                # 📝 수정: BaseTool 상속
│   │   ├── query_refiner.py      # ✅ 유지
│   │   ├── answer_generator.py   # ✅ 유지
│   │   └── models.py              # ✅ 유지
│   ├── rag_retriever/             # 📝 모듈화
│   │   ├── __init__.py
│   │   ├── tool.py                # BaseTool 상속
│   │   ├── retriever.py           # 기존 로직
│   │   └── models.py              # 결과 모델
│   └── legacy/                    # 📦 레거시 보관
│       └── ...
│
├── tasks/
│   ├── __init__.py
│   ├── chat_task.py               # 📝 수정: 개선된 분류
│   ├── react/
│   │   ├── __init__.py
│   │   ├── agent.py               # 📝 개선: 싱글톤 제거
│   │   ├── thought_action_generator.py  # ✅ 유지
│   │   ├── tool_executor.py       # 📝 정리: dead code 제거
│   │   ├── answer_generator.py    # 📝 수정: 중복 제거
│   │   ├── context_manager.py     # 📝 확장: pruning 적용
│   │   ├── verification.py        # 📝 수정: PromptRegistry 사용
│   │   ├── plan_executor.py       # 📝 수정: notepad 지원
│   │   └── utils.py               # ✅ 유지
│   ├── plan_execute.py            # 📝 간소화: 얇은 래퍼
│   ├── smart_agent_task.py        # 📝 수정: 실제 라우팅 로직
│   └── legacy/
│       └── Plan_execute.py        # 📦 레거시 보관
│   # core/agent_graph.py → 삭제 (미사용)
│
├── api/
│   ├── app.py                     # 📝 수정: 미들웨어 추가
│   ├── middleware/                # ✨ 신규 미들웨어
│   │   ├── __init__.py
│   │   ├── request_logging.py
│   │   ├── error_handling.py      # 전역 에러 핸들러
│   │   ├── security_headers.py
│   │   └── rate_limiting.py       # 속도 제한
│   ├── dependencies/              # ✨ 신규 의존성
│   │   ├── __init__.py
│   │   ├── auth.py                # get_current_user 이동
│   │   └── role_checker.py        # RBAC 의존성
│   └── routes/
│       ├── __init__.py            # 📝 수정: 라우터 등록
│       ├── chat.py                # 📝 간소화: 509→300줄
│       ├── auth.py                # 📝 수정: 비밀번호 해싱
│       ├── files.py               # 📝 수정: 검증 추가
│       ├── admin.py               # 📝 수정: RBAC 적용
│       └── tools.py               # ✅ 유지
│
└── storage/
    └── conversation_store.py      # 📝 개선: 쿼리 최적화

tests/                             # ✨ 신규 테스트
├── unit/
│   ├── test_file_handler.py
│   ├── test_prompt_registry.py
│   ├── test_base_tool.py
│   └── test_services.py
├── integration/
│   ├── test_react_agent.py
│   ├── test_api_endpoints.py
│   └── test_file_upload.py
└── prompts/
    └── test_prompt_quality.py     # 프롬프트 검증
```

---

## 구현 계획 (Phase별)

### Phase 1: Core Infrastructure (우선순위: CRITICAL)

**목표**: 중복 제거, 공통 인프라 구축

#### 1.1 통합 파일 핸들러 시스템
```python
# backend/services/file_handler/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

class UnifiedFileHandler(ABC):
    """통합 파일 핸들러 - python_coder와 file_analyzer 공용"""

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """핸들러가 이 파일을 지원하는지 확인"""
        pass

    @abstractmethod
    def extract_metadata(
        self,
        file_path: Path,
        quick_mode: bool = False
    ) -> Dict[str, Any]:
        """메타데이터 추출 (코드 생성용)"""
        pass

    @abstractmethod
    def analyze(self, file_path: Path, query: str = "") -> Dict[str, Any]:
        """전체 분석 (파일 분석 도구용)"""
        pass

    # 공통 유틸리티
    def format_file_size(self, size_bytes: int) -> str:
        """파일 크기 포맷팅 - 단일 구현"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
```

**영향**:
- `python_coder/file_handlers/` 삭제 (7개 파일)
- `file_analyzer/handlers/` 삭제 (7개 파일)
- 약 **600줄 코드 제거**

#### 1.2 BaseTool 인터페이스
```python
# backend/core/base_tool.py
from abc import ABC, abstractmethod
from typing import Optional
from backend.core.result_types import ToolResult
from backend.utils.logging_utils import get_logger
from backend.utils.llm_factory import LLMFactory

class BaseTool(ABC):
    """모든 도구의 기본 인터페이스"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._llm = None

    @abstractmethod
    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """도구 실행 - 표준화된 시그니처"""
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """입력 검증"""
        pass

    @property
    def name(self) -> str:
        """도구 이름"""
        return self.__class__.__name__

    def _get_llm(self, **config):
        """LLM 지연 로딩"""
        if self._llm is None:
            self._llm = LLMFactory.create_llm(**config)
        return self._llm

    def _handle_error(self, e: Exception, context: str) -> ToolResult:
        """표준 에러 핸들링"""
        self.logger.error(f"[{context}] {e}", exc_info=True)
        return ToolResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__
        )
```

#### 1.3 표준 Result 타입
```python
# backend/core/result_types.py
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from datetime import datetime

class ToolResult(BaseModel):
    """모든 도구의 표준 반환 타입"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

#### 1.4 PromptRegistry 통합
```python
# backend/config/prompts/registry.py
from typing import Dict, Callable, Any, List
import inspect
from functools import lru_cache

class PromptRegistry:
    """단일 프롬프트 레지스트리 - utils/prompt_builder.py 대체"""

    _instance = None
    _registry: Dict[str, Callable] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, func: Callable):
        """프롬프트 함수 등록"""
        # 파라미터 검증
        sig = inspect.signature(func)
        cls._registry[name] = {
            'func': func,
            'params': list(sig.parameters.keys()),
            'doc': func.__doc__
        }

    @classmethod
    @lru_cache(maxsize=128)
    def get(cls, name: str, **kwargs) -> str:
        """프롬프트 가져오기"""
        if name not in cls._registry:
            raise ValueError(f"Prompt '{name}' not registered")

        entry = cls._registry[name]
        func = entry['func']

        # 파라미터 검증
        required = [p for p in entry['params']
                    if inspect.signature(func).parameters[p].default == inspect.Parameter.empty]
        missing = set(required) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        return func(**kwargs)

    @classmethod
    def validate_all(cls) -> List[str]:
        """모든 프롬프트 검증"""
        issues = []
        for name, entry in cls._registry.items():
            try:
                # 더미 파라미터로 테스트
                test_params = {p: f"test_{p}" for p in entry['params']}
                prompt = entry['func'](**test_params)

                # 길이 검증
                if len(prompt) > 10000:
                    issues.append(f"{name}: Too long ({len(prompt)} chars)")

                # 필수 요소 확인
                if "You are" not in prompt and "Task:" not in prompt:
                    issues.append(f"{name}: Missing role/task definition")

            except Exception as e:
                issues.append(f"{name}: {e}")

        return issues
```

**변경 사항**:
- `utils/prompt_builder.py` **삭제**
- 모든 모듈이 `from backend.config.prompts import PromptRegistry` 사용
- `verification.py:124` 하드코딩 제거

#### 1.5 보안 강화
```python
# backend/utils/auth.py (수정)
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """비밀번호 해싱"""
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """비밀번호 검증"""
    return pwd_context.verify(plain, hashed)

# backend/api/routes/auth.py (수정)
@auth_router.post("/signup")
async def signup(request: SignupRequest):
    hashed_password = hash_password(request.password)  # ✅ 해싱 적용
    # 사용자 저장 로직
```

---

### Phase 2: Tools 리팩토링

#### 2.1 모든 도구에 BaseTool 적용
```python
# backend/tools/python_coder/tool.py
from backend.core.base_tool import BaseTool
from backend.core.result_types import ToolResult

class PythonCoderTool(BaseTool):
    async def execute(
        self,
        query: str,
        context: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """표준화된 실행 인터페이스"""
        if not self.validate_inputs(query=query):
            return self._handle_error(
                ValueError("Invalid inputs"),
                "input_validation"
            )

        # 기존 로직
        result = await self.orchestrator.execute_code_task(...)

        return ToolResult(
            success=result["success"],
            output=result.get("output"),
            error=result.get("error"),
            metadata={
                "code": result.get("code"),
                "execution_time": result.get("execution_time"),
                "total_attempts": result.get("total_attempts")
            }
        )

    def validate_inputs(self, **kwargs) -> bool:
        query = kwargs.get("query", "")
        return len(query.strip()) > 0
```

#### 2.2 파일 핸들러 통합
```python
# backend/tools/python_coder/generator.py (수정)
from backend.services.file_handler import FileHandlerRegistry

class CodeGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.file_handler = FileHandlerRegistry()  # ✅ 통합 핸들러 사용

    async def generate(self, query: str, file_paths: List[str], ...):
        # 파일 메타데이터 추출
        file_context = []
        for path in file_paths:
            handler = self.file_handler.get_handler(path)
            metadata = handler.extract_metadata(path, quick_mode=False)
            file_context.append(metadata)

        # 프롬프트 생성
        prompt = PromptRegistry.get(
            'python_code_generation',
            query=query,
            file_context=file_context,
            ...
        )
        # 생성 로직
```

#### 2.3 Executor 분할 (843줄 → 모듈)
```python
# backend/tools/python_coder/executor/core.py (~200줄)
class CodeExecutor:
    """핵심 실행 로직만"""

    def __init__(self):
        self.import_validator = ImportValidator()
        self.repl_manager = REPLManager()
        self.sandbox = SandboxConfig()

    async def execute(self, code: str, session_id: str) -> Dict:
        # 실행 로직만
        pass

# backend/tools/python_coder/executor/import_validator.py (~150줄)
class ImportValidator:
    """임포트 검증만"""
    BLOCKED_IMPORTS = {...}

    def validate(self, code: str) -> Tuple[bool, List[str]]:
        # AST 파싱 및 검증
        pass

# backend/tools/python_coder/executor/repl_manager.py (~150줄)
class REPLManager:
    """영구 REPL 관리만"""

    def get_or_create(self, session_id: str) -> Any:
        # REPL 생성/재사용
        pass
```

---

### Phase 3: Agents 리팩토링

#### 3.1 Agent Graph 제거
```bash
# 삭제: backend/core/agent_graph.py (미사용)
# 이유: ToolExecutor와 중복, 사용되지 않음
```

#### 3.2 ReAct Agent 개선
```python
# backend/tasks/react/agent.py (수정)

# ❌ 기존: 전역 싱글톤
# react_agent = ReActAgent(max_iterations=6)

# ✅ 수정: 팩토리 패턴
class ReActAgentFactory:
    @staticmethod
    def create(
        max_iterations: int = 6,
        session_id: Optional[str] = None
    ) -> ReActAgent:
        """요청별 새 인스턴스 생성"""
        agent = ReActAgent(max_iterations=max_iterations)
        agent.session_id = session_id

        # Notepad 로드
        if session_id:
            notepad = SessionNotepad.load(session_id)
            agent.context_manager.set_notepad(notepad)

        return agent

# backend/tasks/chat_task.py (수정)
async def execute_task(query: str, session_id: str, ...):
    if task_type == "agentic":
        # ✅ 요청별 새 에이전트
        agent = ReActAgentFactory.create(session_id=session_id)
        result = await agent.execute(query, ...)
```

#### 3.3 SmartAgent 라우터 수정
```python
# backend/tasks/smart_agent_task.py (수정)

async def execute_smart_agent_task(...):
    # ❌ 기존: 항상 Plan-Execute
    # selected_agent = AgentType.PLAN_EXECUTE

    # ✅ 수정: 실제 휴리스틱
    selected_agent = _select_agent(query, context)

    if selected_agent == AgentType.REACT:
        agent = ReActAgentFactory.create(session_id=session_id)
        return await agent.execute(...)
    else:
        return await execute_plan_execute(...)

def _select_agent(query: str, context: Dict) -> AgentType:
    """실제 라우팅 로직"""
    # 파일이 많거나 복잡한 경우 → Plan-Execute
    if context.get("file_count", 0) > 5:
        return AgentType.PLAN_EXECUTE

    # 다단계 키워드 → Plan-Execute
    multi_step_keywords = ["first", "then", "after that", "finally", "step by step"]
    if any(kw in query.lower() for kw in multi_step_keywords):
        return AgentType.PLAN_EXECUTE

    # 기본: ReAct (빠르고 유연)
    return AgentType.REACT
```

#### 3.4 Dead Code 제거
```python
# backend/tasks/react/tool_executor.py (수정)

# ❌ 삭제: attempted_coder 파라미터 (사용되지 않음)
async def _execute_rag_retrieval(
    self,
    action_input: str,
    session_id: Optional[str] = None,
    # attempted_coder: bool = False,  # ❌ 삭제
) -> str:
    """RAG 검색 실행"""
    # 단순화된 로직
    results = await rag_retriever.retrieve(action_input, ...)
    return self._format_rag_results(results)
```

---

### Phase 4: 프롬프트 최적화

#### 4.1 Python Coder 프롬프트 분할 (328줄 → 모듈)
```python
# backend/config/prompts/python_coder/__init__.py
def get_python_code_generation_prompt(
    query: str,
    context: Optional[str] = None,
    file_context: Optional[List[Dict]] = None,
    **kwargs
) -> str:
    """메인 생성 프롬프트 - 조합 방식"""
    from .generation import get_base_generation_prompt
    from .templates import get_file_context_section, get_rules_section

    parts = [get_base_generation_prompt(query)]

    if file_context:
        parts.append(get_file_context_section(file_context))

    if kwargs.get('plan_context'):
        from .templates import get_plan_section
        parts.append(get_plan_section(kwargs['plan_context']))

    parts.append(get_rules_section())

    return "\n\n".join(parts)

# backend/config/prompts/python_coder/generation.py (~100줄)
def get_base_generation_prompt(query: str) -> str:
    """기본 생성 프롬프트 - 간결화"""
    return f"""You are a Python code generator.

Task: {query}

Generate Python code that:
1. Answers the user's question directly
2. Uses exact filenames shown in file context
3. Includes clear output/visualization
4. Handles errors gracefully

Output only the Python code in markdown:
```python
# Your code here
```
"""

# backend/config/prompts/python_coder/templates.py (~40줄)
def get_file_context_section(files: List[Dict]) -> str:
    """파일 컨텍스트 섹션"""
    lines = ["Available Files:"]
    for f in files:
        lines.append(f"- {f['filename']}: {f['rows']} rows, {len(f['columns'])} columns")
    return "\n".join(lines)

def get_rules_section() -> str:
    """규칙 섹션 - 핵심만"""
    return """CRITICAL RULES:
1. Use EXACT filenames from file list
2. NO generic names (data.json, file.csv)
3. Print or save results
4. NO user input (sys.argv, input())
"""
```

**결과**: 328줄 → ~280줄 구조화 (조합 가능, 테스트 가능)

#### 4.2 Task Classification 개선
```python
# backend/config/prompts/task_classification.py (수정)
def get_agentic_classifier_prompt() -> str:
    return """You are a task classifier. Classify user queries into "agentic" or "chat".

AGENTIC - Requires tools (web search, code execution, file analysis):
✓ "What's the weather in Seoul RIGHT NOW?" (current data)
✓ "Analyze sales_data.csv and calculate mean" (file + code)
✓ "Search for AI developments in 2024" (explicit search)
✓ "Compare Python vs JavaScript performance" (research)

CHAT - Can be answered from knowledge base:
✓ "What is Python?" (general knowledge)
✓ "Explain recursion" (concept)
✓ "How to search files in Linux?" (explanation, not actual search)
✓ "Tell me about the Eiffel Tower" (encyclopedia)

Edge Cases:
- "How to calculate variance?" → CHAT (explain concept)
- "Calculate variance of [1,2,3,4,5]" → AGENTIC (execute)
- "What are recent AI developments?" → AGENTIC (recent = current)
- "What is machine learning?" → CHAT (established)

Respond ONLY: "agentic" or "chat"
"""
```

#### 4.3 하드코딩된 프롬프트 제거
```python
# backend/tasks/react/verification.py (수정)

# ❌ 기존: 하드코딩
# verification_prompt = f"""Verify if the step goal was achieved..."""

# ✅ 수정: PromptRegistry 사용
from backend.config.prompts import PromptRegistry

verification_prompt = PromptRegistry.get(
    'react_step_verification',
    plan_step_goal=plan_step.goal,
    success_criteria=plan_step.success_criteria,
    tool_used=tool_used,
    observation=observation[:1000]
)
```

---

### Phase 5: API Layer

#### 5.1 Service Layer 도입
```python
# backend/services/chat_service.py
class ChatService:
    """채팅 비즈니스 로직"""

    @staticmethod
    async def process_message(
        user_id: str,
        message: str,
        session_id: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        agent_type: str = "AUTO"
    ) -> Dict[str, Any]:
        """메시지 처리 로직"""
        # 작업 분류
        task_type = await classify_task(message)

        # 에이전트 실행
        if task_type == "agentic":
            agent = ReActAgentFactory.create(session_id=session_id)
            result = await agent.execute(message, file_paths=file_paths)
        else:
            result = await simple_chat(message, session_id)

        # 대화 저장
        await ConversationService.save_message(...)

        return result

# backend/api/routes/chat.py (간소화: 509→300줄)
@chat_router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,  # ✅ Pydantic 검증
    current_user: Dict = Depends(get_current_user)
):
    """얇은 컨트롤러"""
    try:
        # Service 호출
        result = await ChatService.process_message(
            user_id=current_user["username"],
            message=request.messages[-1]["content"],
            session_id=request.session_id,
            file_paths=request.file_paths
        )

        # 표준 응답
        return ChatCompletionResponse(
            choices=[...],
            x_session_id=result["session_id"],
            x_agent_metadata=result.get("metadata")
        )

    except Exception as e:
        # 전역 에러 핸들러가 처리
        raise
```

#### 5.2 미들웨어 추가
```python
# backend/api/middleware/error_handling.py
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 에러 핸들링"""
    logger.error(f"Unhandled error: {exc}", exc_info=True)

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=APIError(
                error=exc.detail,
                error_code=f"HTTP_{exc.status_code}",
                timestamp=datetime.utcnow(),
                request_id=request.state.request_id
            ).dict()
        )

    return JSONResponse(
        status_code=500,
        content=APIError(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.utcnow()
        ).dict()
    )

# backend/api/middleware/security_headers.py
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response
```

#### 5.3 RBAC 의존성
```python
# backend/api/dependencies/role_checker.py
def require_role(required_role: str):
    """역할 기반 접근 제어"""
    async def dependency(current_user: Dict = Depends(get_current_user)):
        if current_user.get("role") != required_role:
            raise HTTPException(403, "Insufficient permissions")
        return current_user
    return Depends(dependency)

# backend/api/routes/admin.py (사용)
@admin_router.post("/model")
async def change_model(
    request: ModelChangeRequest,
    current_user: Dict = require_role("admin")  # ✅ 간결한 RBAC
):
    """모델 변경"""
    # 로직
```

---

### Phase 6: 검증

#### 6.1 기능 동등성 테스트
```python
# tests/integration/test_feature_parity.py
import pytest
from backend.services.chat_service import ChatService

class TestFeatureParity:
    """리팩토링 후 기능 동등성 확인"""

    @pytest.mark.asyncio
    async def test_simple_chat(self):
        """간단한 채팅 동작"""
        result = await ChatService.process_message(
            user_id="test_user",
            message="What is Python?",
            session_id=None
        )
        assert result["success"] is True
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_code_generation(self):
        """코드 생성 동작"""
        result = await ChatService.process_message(
            user_id="test_user",
            message="Calculate the sum of [1,2,3,4,5]",
            file_paths=None
        )
        assert "code" in result
        assert "15" in result["output"] or "15" in result["response"]

    @pytest.mark.asyncio
    async def test_file_analysis(self):
        """파일 분석 동작"""
        # 테스트 파일 생성
        test_csv = create_test_csv()

        result = await ChatService.process_message(
            user_id="test_user",
            message="Analyze this CSV and show column names",
            file_paths=[test_csv]
        )
        assert result["success"] is True
```

#### 6.2 중복 제거 확인
```bash
# scripts/check_duplication.sh
#!/bin/bash

echo "=== Checking for duplicate code patterns ==="

# 1. 파일 핸들러 중복 확인
echo "1. File handlers should be unified:"
find backend -name "*_handler.py" | grep -E "(python_coder|file_analyzer)" && echo "❌ FAIL: Still duplicated" || echo "✅ PASS: Unified"

# 2. PromptRegistry 중복 확인
echo "2. PromptRegistry should be single:"
find backend -type f -name "*.py" -exec grep -l "class PromptRegistry" {} \; | wc -l | grep -q "^1$" && echo "✅ PASS: Single registry" || echo "❌ FAIL: Multiple registries"

# 3. 하드코딩된 프롬프트 확인
echo "3. No hardcoded prompts:"
grep -r "f\"\"\".*You are" backend/tasks backend/tools --include="*.py" && echo "❌ FAIL: Hardcoded prompts found" || echo "✅ PASS: No hardcoded prompts"

# 4. PromptRegistry 채택률
echo "4. PromptRegistry adoption rate:"
total=$(find backend -name "*.py" | wc -l)
using=$(grep -r "PromptRegistry.get" backend --include="*.py" | cut -d: -f1 | sort -u | wc -l)
rate=$((using * 100 / total))
echo "Adoption: ${rate}% (target: >50%)"
[[ $rate -gt 50 ]] && echo "✅ PASS" || echo "❌ FAIL"
```

#### 6.3 프롬프트 품질 검증
```python
# tests/prompts/test_prompt_quality.py
from backend.config.prompts import PromptRegistry

class TestPromptQuality:
    def test_all_prompts_registered(self):
        """모든 프롬프트가 등록되었는지 확인"""
        issues = PromptRegistry.validate_all()
        assert len(issues) == 0, f"Prompt issues: {issues}"

    def test_prompt_length_limits(self):
        """프롬프트 길이 제한"""
        for name in PromptRegistry.list_all():
            # 더미 파라미터
            params = {p: f"test_{p}" for p in PromptRegistry.get_params(name)}
            prompt = PromptRegistry.get(name, **params)

            # 10,000자 제한 (약 2500 토큰)
            assert len(prompt) < 10000, f"{name} too long: {len(prompt)} chars"

    def test_no_hardcoded_prompts(self):
        """하드코딩된 프롬프트 없음"""
        import subprocess

        result = subprocess.run(
            ["grep", "-r", "f\"\"\".*You are", "backend/tasks", "backend/tools"],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0, f"Hardcoded prompts found:\n{result.stdout}"
```

---

## 성공 지표 (Success Metrics)

### 코드 품질

| 지표 | 현재 | 목표 | 측정 방법 |
|------|------|------|----------|
| 중복 코드 줄 수 | ~1200 줄 | <200 줄 | `radon` 도구 |
| 평균 파일 크기 | 350 줄 | <250 줄 | `find` + `wc` |
| 500+ 줄 파일 수 | 5개 | 0개 | `find` + `awk` |
| PromptRegistry 채택률 | 16% | >80% | `grep` 분석 |
| 하드코딩 프롬프트 | 3+ | 0 | 정규식 검색 |
| BaseTool 채택률 | 0% | 100% | 상속 체크 |

### 보안

| 지표 | 현재 | 목표 |
|------|------|------|
| 평문 비밀번호 | ❌ | ✅ bcrypt |
| 파일 업로드 검증 | ❌ | ✅ 크기/타입 제한 |
| RBAC 적용 | 1개 엔드포인트 | 모든 admin 엔드포인트 |
| 보안 헤더 | ❌ | ✅ 전체 적용 |

### 성능

| 지표 | 현재 | 목표 |
|------|------|------|
| Python coder 프롬프트 토큰 | ~3000 | <1500 |
| 평균 LLM 호출 수 (코드 생성) | 8-10 | 5-7 |
| 파일 핸들러 초기화 시간 | ~200ms | <50ms (레지스트리 캐싱) |

### 유지보수성

| 지표 | 현재 | 목표 |
|------|------|------|
| 순환 의존성 | 1개 (config↔prompts) | 0개 |
| 테스트 커버리지 | ~10% | >60% |
| 문서화율 | ~40% | >80% |

---

## 위험 관리

### High Risk

1. **파일 핸들러 통합 시 호환성 깨짐**
   - **완화**: 병렬 실행 (legacy 폴더 유지), 단계적 마이그레이션
   - **롤백**: legacy 파일로 즉시 복원 가능

2. **PromptRegistry 통합 시 동작 변경**
   - **완화**: 프롬프트 단위 테스트, A/B 비교
   - **롤백**: 이전 프롬프트 저장

3. **BaseTool 적용 시 에이전트 동작 변경**
   - **완화**: 통합 테스트 전체 실행
   - **롤백**: 도구별 독립적 변경, 점진적 적용

### Medium Risk

4. **Agent 싱글톤 제거 시 성능 저하**
   - **완화**: 에이전트 풀링, 재사용 최적화
   - **모니터링**: 응답 시간 측정

5. **대형 파일 분할 시 임포트 에러**
   - **완화**: `__init__.py`에서 backward compatibility 유지
   - **검증**: CI/CD에서 임포트 테스트

---

## 타임라인

### Week 1-2: Core Infrastructure
- [ ] 통합 파일 핸들러
- [ ] BaseTool 인터페이스
- [ ] PromptRegistry 통합
- [ ] 보안 강화 (비밀번호 해싱)

### Week 3-4: Tools 리팩토링
- [ ] 모든 도구에 BaseTool 적용
- [ ] Executor 분할 (843줄)
- [ ] Orchestrator 분할 (788줄)
- [ ] 파일 핸들러 마이그레이션

### Week 5-6: Agents 리팩토링
- [ ] Agent Graph 제거
- [ ] ReAct 싱글톤 제거
- [ ] SmartAgent 라우터 수정
- [ ] Dead code 제거
- [ ] Context manager 개선

### Week 7-8: 프롬프트 최적화
- [ ] Python coder 프롬프트 분할
- [ ] Task classification 개선
- [ ] 하드코딩 프롬프트 제거
- [ ] 검증 프롬프트 완화

### Week 9-10: API & Services
- [ ] Service layer 도입
- [ ] 미들웨어 추가
- [ ] RBAC 적용
- [ ] 응답 표준화

### Week 11-12: 검증 & 최종화
- [ ] 통합 테스트 작성
- [ ] 성능 벤치마크
- [ ] 문서 업데이트
- [ ] 배포 준비

---

## 다음 단계

1. **승인 받기**: 이 계획 검토 및 승인
2. **브랜치 생성**: `refactor/phase-1-core-infrastructure`
3. **Phase 1 시작**: 통합 파일 핸들러부터
4. **일일 커밋**: 작은 단위로 검증하며 진행
5. **PR 리뷰**: Phase별 PR 생성 및 리뷰

이 계획은 **점진적이고 검증 가능한 방식**으로 전체 코드베이스를 리팩토링합니다.
