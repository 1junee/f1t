import os
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))
from intent_analysis import IntentAnalysisModule

REF_DIR = os.path.join(os.path.dirname(__file__), "ref_images")

TEST_QUERIES = [
    # 2505951 - 티셔츠
    {"id": "2505951_C", "query": "이 티셔츠랑 색이 비슷하며, 신축성이 있는 맨투맨 골라줘.", "image_path": f"{REF_DIR}/2505951.jpg"},
    {"id": "2505951_A", "query": "이 티셔츠와 기장이 비슷한 상의 찾아줘.", "image_path": f"{REF_DIR}/2505951.jpg"},

    # 2080479 - 맨투맨
    {"id": "2080479_C", "query": "이 맨투맨이랑 기장이 비슷하며, 여름에 입을 상의 골라줘.", "image_path": f"{REF_DIR}/2080479.jpg"},
    {"id": "2080479_A", "query": "이 맨투맨과 색이 비슷한 티셔츠 찾아줘.", "image_path": f"{REF_DIR}/2080479.jpg"},

    # 2768893 - 바지
    {"id": "2768893_C", "query": "이 바지랑 기장은 비슷한데, 두께가 얇은 바지 추천해줘.", "image_path": f"{REF_DIR}/2768893.jpg"},
    {"id": "2768893_A", "query": "이 바지랑 색이 비슷한 바지 찾아줘.", "image_path": f"{REF_DIR}/2768893.jpg"},

    # 3276663 - 슬랙스
    {"id": "3276663_C", "query": "이 바지랑 색은 비슷한데, 핏이 보통인 슬랙스 추천해줘.", "image_path": f"{REF_DIR}/3276663.jpg"},
    {"id": "3276663_A", "query": "이 바지보다 기장은 짧은 바지 찾아줘.", "image_path": f"{REF_DIR}/3276663.jpg"},

    # 1202891 - 치마
    {"id": "1202891_C", "query": "이 치마랑 기장은 비슷한데, 신축성이 있는 치마 골라줘.", "image_path": f"{REF_DIR}/1202891.jpg"},
    {"id": "1202891_A", "query": "이 치마랑 색이 비슷한 치마 찾아줘.", "image_path": f"{REF_DIR}/1202891.jpg"},

    # 2948636 - 치마
    {"id": "2948636_C", "query": "이 치마랑 기장은 비슷한데, 얇은 치마 골라줘.", "image_path": f"{REF_DIR}/2948636.jpg"},
    {"id": "2948636_A", "query": "이 치마보다 기장이 긴 치마 찾아줘.", "image_path": f"{REF_DIR}/2948636.jpg"},

    # 1349530 - 치마
    {"id": "1349530_C", "query": "이 치마랑 색은 비슷한데, 와이드 핏의 치마 골라줘.", "image_path": f"{REF_DIR}/1349530.jpg"},
    {"id": "1349530_A", "query": "이 치마랑 비슷한 기장의 치마 찾아줘.", "image_path": f"{REF_DIR}/1349530.jpg"},

    # 1350308 - 원피스
    {"id": "1350308_C", "query": "이 원피스랑 기장은 비슷한데, 겨울에 입을 원피스 골라줘.", "image_path": f"{REF_DIR}/1350308.jpg"},
    {"id": "1350308_A", "query": "이 원피스랑 색이 비슷한 원피스 찾아줘.", "image_path": f"{REF_DIR}/1350308.jpg"},

    # 3639514 - 원피스
    {"id": "3639514_C", "query": "이 원피스랑 색은 비슷한데, 두꺼운 원피스 골라줘.", "image_path": f"{REF_DIR}/3639514.jpg"},
    {"id": "3639514_A", "query": "이 원피스보다 기장이 짧은 원피스 찾아줘.", "image_path": f"{REF_DIR}/3639514.jpg"},

    # 3602675 - 원피스
    {"id": "3602675_C", "query": "이 원피스랑 기장은 비슷한데, 밝은 색 원피스 골라줘.", "image_path": f"{REF_DIR}/3602675.jpg"},
    {"id": "3602675_A", "query": "이 원피스랑 기장이 비슷한 원피스 찾아줘.", "image_path": f"{REF_DIR}/3602675.jpg"},
]


def main():
    # pytest 단위 테스트가 아니라 모델을 직접 실행하고 결과 JSON을 쓰는 독립 실험이다.
    # 운영 /search를 호출하지 않으므로 이 결과만으로 전체 검색 성공을 확인할 수는 없다.
    output_path = os.path.join(os.path.dirname(__file__), "image_test_results.json")

    print("\n" + "="*60)
    print("🚀 Initializing model...")
    print("="*60)
    analyzer = IntentAnalysisModule()

    print("\n" + "="*60)
    print("🧪 Running inference on image test queries")
    print("="*60)

    batch_results = analyzer.infer_batch(TEST_QUERIES)

    for result in batch_results:
        result["inferred_attributes"].pop("loaded_image_url", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, indent=4, ensure_ascii=False)
    print(f"\n💾 Results saved to '{output_path}'")


if __name__ == "__main__":
    main()
