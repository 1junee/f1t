from __future__ import annotations

from pathlib import Path


def load_pipeline_env() -> None:
    """Load pipeline-local environment variables when python-dotenv is installed."""
    # 작업 디렉터리 대신 이 파일의 위치를 기준으로 설정 파일을 찾는다.
    # 반환값은 없고, 이후 os.getenv()가 읽을 프로세스 환경변수를 준비한다.
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional local convenience
        return

    # 로컬 .env를 읽되, 배포 환경에 이미 설정된 환경변수는 덮어쓰지 않는다.
    load_dotenv(Path(__file__).resolve().parent / ".env")
