# AI CUP 2026 圍棋棋力與風格預測：棋力預測與玩家辨識

隊伍：競賽一（玩家辨識）TEAM_11613、競賽二（棋力預測）TEAM_11614，AIdea 帳號 xin101。

本 repository 以官方範例 [AICup-2026-Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial) 為基礎。
兩個子題都先照官方 notebook 跑出 baseline，再各做一個只改推論的小改動。沒有使用預訓練模型或外部資料。

## 結果

| 子題 | 版本 | Public | Private | Private 名次 |
| --- | --- | --- | --- | --- |
| 棋力預測 | baseline（每局機率取平均） | 0.4365169 | - | - |
| 棋力預測 | **最終提交**：每局 log 機率相加 | 0.5087320 | 0.5258305 | 133/175 |
| 玩家辨識 | baseline（平均向量直接比 L2 距離） | 0.3079285 | - | - |
| 玩家辨識 | **最終提交**：重心與查詢向量先 L2 正規化 | 0.3063453 | 0.2754972 | 128/136 |

排行榜以每隊最後一次提交計分，所以最終提交就是小改動版本。baseline 列的 Private 沒有單獨取得，以 `-` 表示。

## 與 baseline 的差異（都只改推論，沒有重新訓練）

- 棋力預測：同一個 `best_model.pth`，把「每局的類別機率取平均」改成「每局 log 機率相加」再取最大。
- 玩家辨識：同一個 `best_model.pth`，把候選玩家的重心向量與每題的查詢向量先做 L2 正規化，再比距離。
- 為了 8 GB 顯卡，棋力預測的 `BATCH_SIZE` 從官方 256 改成 128。
- Python 3.14 的多進程預設是 forkserver，notebook 開頭加了 `mp.set_start_method('fork', force=True)`。

## 環境

- Windows 上的 WSL2 Ubuntu 22.04
- Python 3.14、PyTorch 2.14.1+cu130、webdataset 1.0.2，另需 pandas、numpy、matplotlib、sgfmill、jupyterlab
- ASUS TUF Gaming A17：AMD Ryzen 5 7535HS、32 GB RAM、RTX 4060 Laptop GPU（8 GB VRAM）

```bash
python3 -m venv ~/aicup-env
source ~/aicup-env/bin/activate
# PyTorch 請依 https://pytorch.org 的指令安裝對應 CUDA 版本
pip install pandas numpy matplotlib webdataset sgfmill jupyterlab
```

## 檔案

| 檔案 | 說明 |
| --- | --- |
| `network.py`、`utils.py` | 官方提供的網路架構與 SGF 解析工具 |
| `rank-prediction.ipynb` | 棋力預測：特徵抽取、訓練、推論 |
| `player-identification.ipynb` | 玩家辨識：特徵抽取、訓練、推論，含新增測試集的合併（1,600 列） |
| `submission_rank.csv` | 棋力預測 baseline 提交檔（800 列） |
| `submission_rank_logsum.csv` | 棋力預測最終提交檔（800 列） |
| `submission_player_v2.csv` | 玩家辨識 baseline 提交檔（1,600 列） |
| `submission_player_v2_norm.csv` | 玩家辨識最終提交檔（1,600 列） |

訓練資料、特徵檔與模型權重太大，沒有放進 repository。

## 重現步驟

1. 取得官方程式與資料：
   ```bash
   git clone https://github.com/AILAB-NDHU/AICup-2026-Tutorial
   cd AICup-2026-Tutorial
   ```
   把官方發佈的訓練資料 `training/` 和測試資料 `tests/` 放在專案根目錄，再用本 repository 的檔案覆蓋同名檔案。
2. 啟動 Jupyter：
   ```bash
   source ~/aicup-env/bin/activate
   jupyter lab --no-browser
   ```
3. 棋力預測：開啟 `rank-prediction.ipynb`，由上到下依序執行（特徵抽取 → Data Loader → 訓練 → 推論），
   產生 baseline 的 `submission_rank.csv`，再執行「log 機率相加」的推論 cell 產生 `submission_rank_logsum.csv`。
4. 玩家辨識：開啟 `player-identification.ip
