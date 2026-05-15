import pandas as pd
import yaml
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

# 載入模型 (建議與 Step 4 使用同一顆，確保一致性)
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 載入配置
with open('keywords.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 計算各類別的語義中心點 (Centroids)
class_centroids = {}
for cat_id, cat_info in config['categories'].items():
    if 'coarse_keywords' in cat_info and cat_info['coarse_keywords']:
        # 將關鍵字轉為向量並平均，代表該類別的核心語義
        kw_embeddings = embedder.encode(cat_info['coarse_keywords'])
        class_centroids[cat_id] = np.mean(kw_embeddings, axis=0)

# 定義標籤映射
ABSTAIN = -1
NORMAL = 0
LOAN = 1
PORN = 2
GAMBLING = 3
SCAM = 4

label_map = {
    "loan": LOAN,
    "porn": PORN,
    "gambling": GAMBLING,
    "scam_recruitment": SCAM
}

# --- 規則 A：強特徵命中 (不需上下文) ---
@labeling_function(resources=dict(cfg=config))
def lf_strong_hit(x, cfg):
    text = x['name']
    for cat_id, cat_info in cfg['categories'].items():
        rules = cat_info.get('fine_grained_rules', {})
        strong_words = rules.get('strong_hits', [])
        if any(w in text for w in strong_words):
            return label_map.get(cat_id, ABSTAIN)
    return ABSTAIN

# --- 規則 B：弱特徵與專屬觸發詞 (處理 [LOC] 與歧義) ---
@labeling_function(resources=dict(cfg=config))
def lf_weak_hit_with_context(x, cfg):
    text = x['name']
    for cat_id, cat_info in cfg['categories'].items():
        rules = cat_info.get('fine_grained_rules', {})
        logic = rules.get('weak_hit_logic', {})
        if not logic:
            continue

        targets = logic.get('targets', [])
        triggers = logic.get('triggers', [])
        
        if any(t in text for t in targets) and any(tr in text for tr in triggers):
            return label_map.get(cat_id, ABSTAIN)
    return ABSTAIN

# --- 規則 C：語義距離規則 (Embedding-based Rescue) ---
@labeling_function(resources=dict(centroids=class_centroids, model=embedder))
def lf_embedding_similarity(x, centroids, model):
    text = x['name']
    # 這裡傳入的 text 已經是在 Step 2 處理過包含 [LOC] 的文字
    query_vec = model.encode([text])[0]
    
    best_cat = ABSTAIN
    max_sim = 0.85  # 設定較高的門檻，避免過度泛化
    
    for cat_id, centroid_vec in centroids.items():
        sim = cosine_similarity(query_vec.reshape(1, -1), centroid_vec.reshape(1, -1))[0][0]
        if sim > max_sim:
            max_sim = sim
            best_cat = label_map.get(cat_id, ABSTAIN)
    return best_cat


# 在載入配置後，加入這段
print("正在載入人工審核資料作為標註錨點...")
df_reviewed = pd.read_csv('gold_candidates_reviewed.csv').dropna(subset=['text'])

# 建立 文本 -> 類別名稱 的映射
# 假設人工標記的欄位是 'proposed_category'
reviewed_mapping = dict(zip(df_reviewed['text'], df_reviewed['proposed_category']))

# --- 規則 D：地面真值規則 (Ground Truth) ---
@labeling_function(resources=dict(mapping=reviewed_mapping))
def lf_ground_truth(x, mapping):
    text = x['name']  # 這裡的 x['name'] 應該是已經過 Step 2 [LOC] 處理過的
    
    if text in mapping:
        label_name = mapping[text]
        # 回傳對應的 Label ID
        return label_map.get(label_name, ABSTAIN)
    
    return ABSTAIN


# 準備執行環境
lfs = [lf_strong_hit, lf_weak_hit_with_context, lf_embedding_similarity, lf_ground_truth]
df_train = pd.read_csv('candidate_pool_sanitized.csv')

# 執行標註
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

# 分析效能 (檢視 lf_summary)
print("\n--- 標註函數效能分析 ---")
print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

# 訓練生成模型 (LabelModel)
print("\n訓練 Snorkel LabelModel 中...")
label_model = LabelModel(cardinality=5, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)

# 產出最終標籤
df_train['snorkel_label_idx'] = label_model.predict(L=L_train, tie_break_policy="abstain")
df_train['snorkel_label'] = df_train['snorkel_label_idx'].map({
    NORMAL: "正常困難負樣本",
    LOAN: "借貸融資",
    PORN: "黃色與特種行業",
    GAMBLING: "博弈",
    SCAM: "詐騙高風險招募",
    ABSTAIN: "棄權未分類"
})

# 儲存有效樣本
df_labeled = df_train[df_train['snorkel_label_idx'] != ABSTAIN].copy()
df_labeled.to_csv('snorkel_labeled_training_data.csv', index=False, encoding='utf-8-sig')

print(f"\n✅ 標註完成！最終有效樣本數: {len(df_labeled)}")
print(df_labeled['snorkel_label'].value_counts())


