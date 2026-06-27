# 🏃‍♂️ TCX Batch Parser: Universal Heart Rate Data Extractor

A robust, minimal, and professional Python tool to batch-convert raw `.tcx` training files into clean, analysis-ready CSV datasets.

## 🚀 Key Features
- **Universal Timestamps**: Every data point is exported with Human-Readable Time, Unix Epoch, and Relative Seconds.
- **Batch Processing**: Process hundreds of files in seconds.
- **Auto-Naming**: Output files are automatically named by session date and duration (e.g., `2026-05-11_51min_parsed.csv`).
- **Data-Science Ready**: Output is perfectly formatted for direct import into Pandas, Excel, or SQL.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tcx-batch-parser.git
   cd tcx-batch-parser
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy
   ```

## 📂 Usage
1. Place your raw `.tcx` files in the `input/` folder.
2. Run the parser:
   ```bash
   python tcx_parser.py
   ```
3. Find your cleaned data in the `output/` folder.

## 📊 Output Format
| datetime | unix_epoch | relative_seconds | heart_rate |
| :--- | :--- | :--- | :--- |
| 2026-05-11 08:25:24 | 1778497524 | 0.0 | 87 |

---
*Created for athletes and data analysts who want to own their data.*
