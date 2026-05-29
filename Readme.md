<p align="center">
    <img src="./assets/gif/click.gif" alt="demo click" width="50%">
</p>

# プラグイン情報

[![Release](https://img.shields.io/github/v/release/panko200/SAM3CutoutPlugin)](https://github.com/panko200/SAM3CutoutPlugin)
[![Downloads](https://img.shields.io/github/downloads/panko200/SAM3CutoutPlugin/total)](https://github.com/panko200/SAM3CutoutPlugin/releases/latest)
[![License](https://img.shields.io/github/license/panko200/SAM3CutoutPlugin)](https://github.com/panko200/SAM3CutoutPlugin/blob/master/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/panko200/SAM3CutoutPlugin)](https://github.com/panko200/SAM3CutoutPlugin/commits/master)

SAM3 被写体切り抜きプラグイン (SAM3CutoutPlugin)  
製作者：panko200  
配布場所：[https://github.com/panko200/SAM3CutoutPlugin](https://github.com/panko200/SAM3CutoutPlugin)

# 概要

YukkuriMovieMaker4 にて動作するプラグインです。  
**画像・動画**に対して、物体を**簡単に切り抜く**ことができるようになるプラグインを作成しました。

なお、内部でSAM3というAIモデルを用いていますが、これは学習済みのモデルですので、入力した画像が勝手に学習されたりはしないので、安心してください。  
また、ローカル実行ですので、CPUまたはGPUのパワーが必要です。

詳細な導入・解説につきましては、以下のサイト・動画で解説しています。是非ご参照ください。

- [Note](dammy)
- [Youtube](dammy)
- [niconico](dammy)

# 使用方法

0. 【推奨】 NVIDIA GPUを使用している方は、CUDAのバージョンが、v12.8以上のドライバーをインストールする。
1. Hugging Faceでアカウント作成
2. SAM3の利用許可の認証
3. アクセストークンの発行
4. プラグインをインストールして YMM4 を起動する。
5. `設定` から「SAM3 被写体切り抜き設定」を開く。
6. トークンを入力して、python環境構築
7. `ツール`から、「SAM3 被写体切り抜き」を開く。
8. 各設定を各自で設定し、出力。

# アンインストール方法

1. YMM4 を起動して `ヘルプ(H)` > `その他` > `プラグインフォルダを開く` をクリックする。
2. YMM4 を終了する。
3. `SAM3CutoutPlugin` という名前のフォルダを削除する。
4. `win + R`を押して、`%LOCALAPPDATA%`と入力し、中にある`YMM4_SAM3CutoutPlugin`フォルダを削除する。(フォルダパスは、`C:\Users\(ユーザー名)\AppData\Local\YMM4_SAM3CutoutPlugin`)

# 注意点

OS : Windows11 (64bit)  
ゆっくりMovieMaker4 : v4.52.0.8  
CPU : Ryzen 7 9700X  
GPU : NVIDIA Geforce RTX 4070 Ti  
RAM : DDR5 32GB x2 (64GB)  
にて動作確認をしています。

### 運用上の注意

- **Cドライブの容量を約7GB使用します** 本プラグインは、`%LOCALAPPDATA%` 内に専用のPython環境を構築するため、Cドライブの容量を消費します。完全に削除したい場合は、上記のアンインストール手順に従って `YMM4_SAM3CutoutPlugin` フォルダを `Shift + Del` などで完全に削除してください。

- **GPU（グラフィックボード）の推奨** CPUでの推論も可能ですが処理に非常に時間がかかるため、動画への実行は現実的ではありません。NVIDIA GPUを使用している方はCUDA v12.8以上のドライバーの導入を強く推奨します。**AMD RadeonやIntel Arcをご使用の場合は、CPU推論のみ**となります。

- **入れたらすぐ動くプラグインではありません** 少し複雑な導入工程が必要となります。お手数ですが、解説動画やマニュアルの手順をしっかり踏んで導入を行ってください。

- **動画切り抜き時のメモリ消費について** 動画を切り抜く場合は非常に多くのRAMを消費するため、**最大でも10秒程度**の素材に留めてください。RAMが32GB未満の場合、5秒程度の動画でも処理が厳しくなる可能性があります。

- **プロセスが残ってしまった場合** 処理の途中で強制終了するなどした場合、正常に終了せず、裏でプロセスが常駐してしまうことがあります。動作が重い・おかしいと感じたときは、タスクマネージャーから `python.exe` を探して強制終了してください。

_免責事項: 作者は、本プラグインの使用または使用不能に起因するいかなる損害についても、一切の責任を負いません。_

# アップデート内容

v0.1.0  
公開

# ライセンス

本プラグインは、以下のAviutl2用のプラグインを参考に制作されました。プラグインの開発者に深く感謝申し上げます。

**sam3_bb_gb_generator**

- License: MIT License
- Author: clean262
- URL: [https://github.com/clean262/sam3_bb_gb_generator](https://github.com/clean262/sam3_bb_gb_generator)
- [MIT License](./sam3_bb_gb_generator_LICENSE)

本プロジェクトは、MIT License のもと公開しています。  
[MIT License](./LICENSE)
