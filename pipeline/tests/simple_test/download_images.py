import os
import json
from pathlib import Path
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

# ID -> 조회할 테이블 매핑
ID_TABLE_MAP = {
    2505951: "musinsa_top_clothes",   # 티셔츠
    2080479: "musinsa_top_clothes",   # 맨투맨
    2768893: "musinsa_pants",         # 바지
    3276663: "musinsa_pants",         # 슬랙스
    1202891: "musinsa_skirt_dress",   # 치마
    2948636: "musinsa_skirt_dress",   # 치마
    1349530: "musinsa_skirt_dress",   # 치마
    1350308: "musinsa_skirt_dress",   # 원피스
    3639514: "musinsa_skirt_dress",   # 원피스
    3602675: "musinsa_skirt_dress",   # 원피스
}

SAVE_DIR = os.path.join(os.path.dirname(__file__), "ref_images")
os.makedirs(SAVE_DIR, exist_ok=True)


def fetch_image_url(item_id: int, table: str) -> str | None:
    # DB에서는 이미지 주소만 읽는다. 이미지 바이트 다운로드는 다음 함수가 담당한다.
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY 환경변수가 설정되지 않았습니다.")

    url = f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{item_id}&select=id,image_url"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        print(f"  ❌ Not found: id={item_id} in {table}")
        return None
    return data[0].get("image_url")


def download_image(image_url: str, save_path: str) -> bool:
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return False


def main():
    # 실험용 참고 이미지를 로컬에 저장하고 상품 ID → 저장 경로 목록을 만든다. DB는 수정하지 않는다.
    id_to_path = {}

    for item_id, table in ID_TABLE_MAP.items():
        print(f"[{item_id}] Fetching from {table}...")
        image_url = fetch_image_url(item_id, table)
        if not image_url:
            continue

        ext = image_url.split("?")[0].rsplit(".", 1)[-1] or "jpg"
        save_path = os.path.join(SAVE_DIR, f"{item_id}.{ext}")

        print(f"  Downloading {image_url[:70]}...")
        if download_image(image_url, save_path):
            print(f"  ✅ Saved: {save_path}")
            id_to_path[item_id] = save_path
        else:
            id_to_path[item_id] = None

    # 결과 저장
    result_path = os.path.join(SAVE_DIR, "image_paths.json")
    with open(result_path, "w") as f:
        json.dump(id_to_path, f, indent=2)
    print(f"\n💾 Image paths saved to {result_path}")


if __name__ == "__main__":
    main()
