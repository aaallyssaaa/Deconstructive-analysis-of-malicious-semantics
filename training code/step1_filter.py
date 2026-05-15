import yaml
import pandas as pd
import re
import time

def build_regex_patterns(yaml_path):
    """讀取 YAML 並將各類別的關鍵字編譯為 Regex 樣式"""
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    patterns = {}
    for cat_key, cat_data in config['categories'].items():
        if 'coarse_keywords' in cat_data and cat_data['coarse_keywords']: 
            # 組合 Regex，例如 "貸款|信貸|借款"
            patterns[cat_key] = '|'.join(map(re.escape, cat_data['coarse_keywords']))
    return patterns

def run_coarse_filter(input_csv, output_csv, normal_output_csv, yaml_path, text_column, chunk_size=200000):
    print(f"🚀 開始粗篩: 讀取 {input_csv} 中的 '{text_column}' 欄位...")
    start_time = time.time()
    
    # 1. 取得 Regex 規則
    patterns = build_regex_patterns(yaml_path)
    print(f"載入類別規則: {list(patterns.keys())}\n" + "-"*40)
    
    first_chunk = True
    total_candidates = 0
    total_normal = 0
    total_processed = 0
    
    # 2. 批次讀取資料
    # 使用 usecols 可以確保即便 CSV 很大，我們也只載入需要的欄位和其他必要資訊
    # 若你需要保留所有原始欄位，請將 usecols 參數移除
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        
        # 確保目標欄位存在且為字串型態 (處理 NaN 避免報錯)
        if text_column not in chunk.columns:
            raise ValueError(f"找不到指定的欄位: {text_column}")
            
        chunk[text_column] = chunk[text_column].astype(str)
        
        # 建立一個全為 False 的遮罩
        trigger_mask = pd.Series(False, index=chunk.index)
        
        # 針對每個類別進行 Regex 掃描
        for cat_key, pattern in patterns.items():
            # 新增 boolean 欄位，紀錄該筆資料是否觸發此類別
            chunk[f'matched_{cat_key}'] = chunk[text_column].str.contains(pattern, na=False)
            # 更新總遮罩 (只要觸發任何一個類別就是 True)
            trigger_mask = trigger_mask | chunk[f'matched_{cat_key}']
        
        # 篩選出候選樣本
        candidates = chunk[trigger_mask].copy()

        # normal
        normal = chunk[~trigger_mask].sample(frac=0.00025, random_state=42)
        
        # 3. 寫入結果至新的 CSV
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        candidates.to_csv(output_csv, mode=mode, header=header, index=False)
        normal.to_csv(normal_output_csv, mode=mode, header=header, index=False)
        
        total_processed += len(chunk)
        total_candidates += len(candidates)
        total_normal += len(normal)
        first_chunk = False
        
        print(f"🔄 掃描進度: {total_processed:,} 筆 | 累積抓出: {total_candidates:,} 筆候選")

        
    print("-" * 40)
    print(f"✅ 粗篩完成！總耗時: {time.time() - start_time:.2f} 秒")
    print(f"🎯 最終候選池大小: {total_candidates:,} 筆 (已儲存至 {output_csv})")
    print(f"🎯 正常樣本數量: {total_normal:,} 筆")

# ==========================================
# 執行區 
# ==========================================
if __name__ == "__main__":
    run_coarse_filter(
        input_csv='nodes_page_1hop.csv',       # 你的原始檔案
        output_csv='candidate_pool.csv',       # 輸出的候選池檔案
        normal_output_csv='normal_pool_sample.csv',
        yaml_path='keywords.yaml',             # 字典檔
        text_column='name',                    # 指定分析第三個欄位 "name"
        chunk_size=200000                      # M3 Max 可以輕鬆吃下 20萬筆的 chunk
    )