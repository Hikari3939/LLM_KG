import os
import json
from my_packages.ScrapeStrokeAndSymptoms import scrape_stroke_and_symptoms


if __name__ == "__main__":
    results, report = scrape_stroke_and_symptoms()

    # 项目根目录下的 data 文件夹
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存爬取的数据
    results_path = os.path.join(output_dir, "example.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"文件已保存到 {output_dir} 目录下。")