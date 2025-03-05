# 數位語音助理系統之實作與實現

* Title : Implementation-and-Realization-of-A-Digital-Voice-Assistant-System
* Author : Chien-Hung Lin

## 摘要 (Abstract)
本專案旨在採用最新的深度學習技術來實作一套人工智慧語音助理，結合大規模語音數據集進行訓練，以提高系統的適應性和普遍化能力。專案將針對以下幾個方面進行開發（如圖所示）：數位訊號處理，通過先進的數位訊號處理技術提高語音訊號的清晰度和質量，確保語音資料的準確性和可用性；語者辨識，開發能夠準確辨識不同語者的技術，以提高系統的安全性和個性化服務能力；語音辨識，利用深度學習技術訓練語音辨識模型，並不斷優化模型參數以提高辨識精度；自然語言處理，結合語音與文本的多樣形態進行學習，提升系統的綜合理解能力，讓語音助理能夠更加準確地理解和回應使用者的需求；語音合成，研發高品質的語音合成技術，使語音助理能夠生成自然流暢的語音回應，增強人機互動的自然性。此外，本論文計畫將語音助理整合至嵌入式系統，使其能應用於更多的設備及環境中，通過嵌入至更小且高效的硬體平台上，以實現語音技術於智慧家居、醫療輔助、教育、汽車電子等多個領域的更廣泛應用。

![image](./voice_assistance.png)

## 環境 (Requirements)
* Ubuntu 24.01 LTS
* Docker
* Python 3.10

## (程式)資歷夾中具以下兩種資料夾 ：
- 數位訊號處理(speech_enhancement)      :  通過數位訊號處理提高語音訊號的清晰度和質量，確保語音資料的準確性和可用性。
- 語者辨識(speaker_recognition)        : (語者註冊、語者辨識),開發能夠準確辨識不同語者的技術，以提高系統的安全性和個性化服務能力。
- 語音辨識(automatic_speech_recogni)   : (聲音轉文字),利用深度學習技術訓練語音辨識模型，並不斷優化模型參數以提高辨識精度。
- 自然語言處理(text_generation, LLM)   : (文字生成並回復),結合語音與文本的多樣形態進行學習，提升系統的綜合理解能力。
- 語音合成(text_to_speech)  : (文字轉聲音),研發高品質的語音合成技術，使語音助理能夠生成自然流暢的語音回應。
  

## (ros_robotarm_objdetect)資歷夾中具以下檔案 ：
### 1. 程式 ：
- **Multi-execute.sh**                        ：  啟動下列程式碼，此程式中具備兩種模式，(1)test為確認相機視角是否正確，(2)voice為本專題的主體使用模式，因此使用者須自行編輯程式碼來添加、去除註解字元(#)。
- **pose_action_client_finger_cartesian.py**  ：  驅動協作型機器手臂程式。
- **voice_detect.py**                         ：  偵測語音程式。
- **voice_object_detect.py**                  ：  座標轉換程式(語音辨識版本)。
- **camera_tf_broadcaster.py**                ：  定義相對座標程式。
- **object_detect.py**                        ：  座標轉換程式(點擊螢幕版本)。

### 2. 音檔 ：

  
## 操作流程：


## 引文(Citation)：
Please cite the following if you make use of the code.

>@inproceedings{kye2020meta,
  title={Implementation-and-Realization-of-A-Digital-Voice-Assistant-System},
  author={Chien-Hung Lin},
  year={2024}
}
