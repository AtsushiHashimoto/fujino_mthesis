# 必要なライブラリ

- numpy
- scipy
- scikit-learn
- matplotlib
- pandas
- h5py 
- Pillow (もしかしたらいらないかも...)
- MYSQL-python
	- https://github.com/farcepest/MySQLdb1
- selectivesearch
	- https://github.com/AlpacaDB/selectivesearch
- outlier-robust-spectral-clustering (0.0.1)
	- https://github.com/minoh-lab/outlier_robust_spectral_clustering.git (private)
- caffe 
	- https://github.com/BVLC/caffe
- py-faster-rcnn 
	- https://github.com/rbgirshick/py-faster-rcnn


# external
- ontology
 - 料理オントロジー(http://www.ls.info.hiroshima-cu.ac.jp/cgi-bin/cooking/wiki.cgi)
 - http://www.ls.info.hiroshima-cu.ac.jp/cooking/ontology.zip 
- IMG
 - 調理過程画像 (IMG/step)
- NLP
 - クックパッドデータからレシピ用語間の完全グラフが構築されたデータ
 - http://plata.ar.media.kyoto-u.ac.jp/how-to/recipe-NLP/
- RECIPE
 - クックパッドデータ (mysqlに登録されている前提)
- cookpad\_step\_grep\_recipesp\_norm\_sentsp.Viob2
 - 森研究室から提供
 - クックパッドデータをパースしたもの(iob2形式)
 - 1.86GBあるテキストファイル
- VGG16 
 - 実行手順->初期学習->前準備参照


# 実行手順
- 開発関係の違いからロケールの変更が必要．
	- Ubuntu
		- $ apt-get install language-pack-ja
		- $ update-locale LANG=ja_JP.UTF-8
- 各スクリプトはscriptディレクトリをカレントディレクトリとして実行.
- 修論時の結果はexp/master\_thesisに保存.  
特に修論時のモデルを使用する場合はexp/master\_thesis/caffe\_db\_master\_thesis/Readme.mdも参照.

## 食材のグルーピング
### 実行ファイル 
script/1\_grouping.sh

### 出力
- exp/grouping/
 - cluster\_\*.txt:  
レシピに同時に現れにくいクラスタごとの食材とレシピへの登場回数
 - th\_for\_cluster\_no\__(%.2e)\__(%d).png:  
横軸:クラスタ数 縦軸:許容する共起確率 ファイル名の(%.2e)と(%d)は共起確率とクラスタ数の結果
 - viob2\_cooccurence\_(Tag).pickle:  
途中経過
 - clustering\_keywords.pickle:  
最終的にクラスタリングの対象となった用語

(注)修論時の結果の再現が難しいので, 結果はexp/grouping\_master\_thesisに
 - 大きな違いは修論時には調理器具や動作などを含めてクラスタリングしてしまっていること
 - 共起しない食材は別のグループになっているので調理器具や動作が混ざっていても結果としては問題なく, じゃがいも・豆腐の実験をしてしまっていたことからこのまま進めた

## フローグラフの推定およびラベルの推定
### 実行ファイル
script/2\_flowgraph.sh
- (注)食材の設定が必要(スクリプト内の変数FOODSに直接書き込み: 各食材は料理オントロジーに一致するように表記し, 食材はカンマ区切り). グルーピングの結果(cluster\_\*.txt)から手動で設定. 
- (注)クックパッドデータを登録したmysqlの設定が必要(スクリプト内の変数に直接書き込み)  
	- MYSQL\_USR: ユーザ名,
	- MYSQL\_PASSWORD: パスワード, 
	- MYSQL\_SOCKET: ソケットのパス(mysqld.sockみたいなファイル),  
	- MYSQL\_HOST: ホスト名(例:localhost), 
	- MYSQL\_DB: データベース名(external/RECIPE/ckpd\_data\_spec.pdfに従ってmysqlに登録していればcookpad\_data)
- (注)external/NLP内でどのレシピIDがどのディレクトリにあるかを保存したjsonファイル(exp/flowgraph/recipe\_directroy.json)を作るコードが紛失しまったので, 必要なら作り直してください(NLPディレクトリの中身が変わらなければ作り直す必要はないです).
辞書型でkeyはレシピID, valueはディレクトリ名(例:"0000")です.
	  {
	  "000005f43fc4c95a43eb061be8d06fadc205bbfe": "0000", 
	  "00001944477645056f44a8cc3472d033b4cb3ffa": "0000", 
	  "000021be4ed94ff1875fe6415e98931104ff0e85": "0000", 
	  "000030f290e656d9f7bc5da1fda6fc58f2e4c093": "0000", 
	  "000043699ce77396edbc95666c1614c5232a1965": "0000", 
	  "00004a55b0607a15e7607f7b547513d789961fb7": "0000", 
	  "00006db396ab35b97fad1b0abf8a86ffe7ee2fb6": "0000", 
	  "000078f77917a0f9f316418411eba17016bfcff3": "0000", 
	  "00007926af3af56b30f9ce8229ee8ed38829540d": "0000", 
	  ...

### 出力
- exp/recipe\_location\_steps.json:  
各レシピIDの画像が存在するディレクトリと手順番号の辞書(ディレクトリはレシピIDの先頭文字と思われるが確証はないので一応保存). 
keyはレシピID, valueは"dir"(レシピID)と"steps"(手順番号のリスト).
- exp/recipes/recipes\_(食材名).tsv:  
各行に対象食材を含むレシピのIDを書いたテキストファイル. 
recipes\_今日の料理.tsvは陰性訓練データ用
(クックパッドのカテゴリは階層構造になっている. 
「今日の料理」は最上位から2番目の層にあるカテゴリであらゆる料理を含む. 
最上位のカテゴリ「カテゴリ」にすると「もどきレシピ」や「その他」など, どのようなレシピが入っているかわからないものがあったので避けた
[参考:https://cookpad.com/category/1]).
- exp/flowgraph/flow/(dir-no)/(レシピID).csv:  
フローグラフを表すcsvファイル. 
各行は頂点を表す. 
各列は次の通り.  
dir-noはexternal/NER/data内と同じ構成. 
	- ID:  頂点を識別するID 
	- char-position: (手順番号)-(手順内の文章の番号)-(一文内の単語の番号) 
	- NEtype: レシピ用語のタグ
	- wordseq: レシピ用語
	- edges:  対象の行を終点とする辺を(始点の頂点ID:辺のコスト)で表記し辺が複数ある場合はスペース区切りとなっている.
IDが0の行のedgesが1:0.5 2:0.3だとすると, 1->0というコスト0.5の辺と2->0というコスト0.3の辺の2つがある. 
- exp/flowgraph/label(dir-no)/(レシピID)\_(手順番号).json:  
各画像のラベルを表すファイル. 
辞書型でkeyがレシピ用語のID, valueがそのレシピ用語のラベルを表す辞書. 
ラベルを表す辞書の意味は次の通り.
	- rNE:  レシピ用語. 文章に現れた通りの食材名
	- NEtype:  レシピ用語のタグ. 今回は食材のFと食材の状態を表すAc, Sfを保持.
	- ontology: rNEを料理オントロジーにより表記ゆれをなくした場合の食材名. オントロジーに含まれなければnull.
	- step, sentence, word\_no: それぞれフローグラフのchar-positionに対応
	- weight: エッジ間の重みなどを使う可能性を考えて一応入れておいただけ. 今回は全て1. この値は後のどのコードでも使用していない. 
	- Ac, Sf: NEtypeがFのときのみ存在. 
NEtypeがAcまたはSfの頂点のリストである. 
食材の状態を表すAc, Sfと食材がフローグラフ上でどれだけ近いかを表す. 
頂点はid, hop, weightの3つで表す. 
idがレシピ用語のIDを表す. 
hopが食材の頂点から食材状態の頂点への経路上にあるエッジ数であり, 
正の値であれば食材から食材状態までグラフを辺の方向通りにたどる経路, 
負の値であれば食材から食材状態までグラフを遡る経路となる. 
weightは今回使用しておらず全て1である. 

## データベースの作成 
### 実行ファイル
script/3\_generate\_db.sh
- (注)スクリプトは修論時の実験の例のためにexp/grouping\_master\_thesis,exp/recipes\_master\_thesisを入力にしているところがあるが本来はexp/grouping,exp/recipes. 
そのまま動かすと上書きされるかもしれないので注意
(バックアップはlog/fujino\_master\_thesis. 
元のexp/recipes\_master/thesisと全く同じものを保存している).
- (注)食材のグルーピング時に出力されたクラスタ番号およびそのクラスタに含まれる食材の設定が必要
(それぞれスクリプト内の変数CLUSTER, FOODSに直接書き込み).
- (注) スクリプトでは動作確認用にexp/caffe\_dbではなくexp/caffe\_db\_testを出力先として設定. 
動作確認及び出力例として作っため, 中のデータはあまり正確ではないので注意
(caffemodelはcaffe\_db\_test下のデータベースを用いて学習した結果ではないなど).

### 出力
- exp/selective\_search  
SelectiveSearchの結果を保存. 
ディレクトリ構成はexternal/IMG/stepと同じ. 
遅いので2回目以降は計算を省略し, ここから結果を呼び出して, これ以外の出力データの計算を行う. 
- exp/recipes/(食材名)/recipes\_train.tsv:  
各行に学習に用いるレシピIDを書いたファイル 
- exp/recipes/(食材名)/recipes\_test.tsv:  
各行にテストに用いるレシピIDを書いたファイル 
- exp/recipes/neg(クラスタ番号)/recipes\_train.tsv:  
各行に陰性訓練データとして用いるレシピIDを書いたファイル(クラスタ番号に含まれる食材は含まないレシピID) 
- exp/recipes/neg(クラスタ番号)/recipes\_test.tsv:  
各行にテストに用いるレシピIDを書いたファイル(クラスタ番号に含まれる食材は含まないレシピID) 
- exp/recipes/TEST:  
今まで用いた全ての食材のexp/recipes/(食材名)/recipes\_test.tsvをコピーしたディレクトリ. 
訓練データに用いるレシピIDがrecipes\_test.tsvと重複しないようにするため,
このディレクトリ内のrecipes\_test.tsvに書かれているレシピIDは今後作られるrecipes\_train.tsvには使われない. 
また, マルチラベルクラス分類の評価をすることを想定すると後に1枚の画像に対して別のグループの食材のアノテーションを行う必要がある. 
アノテーションはグループごとに行う(後述: 評価->前準備)ので, 
複数のグループのアノテーションが行われた画像を少しでも増やすため, 
このディレクトリ内のrecipes\_test.tsvに書かれているレシピIDは今後作られるrecipes\_test.tsvに優先して使われる
(修論では結局マルチラベルクラス分類の評価はしていないのでこの仕様はいらなかった). 
- exp/caffe\_db/(クラスタ番号)/init: 
	- data\_\*.npy:  
データ数\*3\*画像サイズ\*画像サイズのnumpy.array. 学習に用いる部分画像. ひとつのhdf5ファイルが一定以上大きくできないというcaffeの仕様上もちいるデータを複数のデータベースに分けている.
	- img\_list\_\*.tsv:  
部分画像の詳細. 各列は以下の通り. 順番は同じsuffixのdata\_\*.npyと対応.
		- path:  
切り出す前の画像のパス 
		- resize\_w, resize\_h:  
SelectiveSearchをかけるために元の画像を一旦リサイズしたときの大きさ
(元のサイズのままSelectiveSearchをするととても遅い. 224\*224でも数秒かかる). 
		- x, y, w, h:  
切り出した矩形領域の**リサイズ後**の画像上での左上の座標と大きさ
		- labels:  
画像につけられたラベル
		- \_\_background\_\_,　食材1,  食材2  … : 
背景クラスと対象クラスタに含まれる食材を表す. 
各食材がラベルに含まれていれば1.
そうでなければ0.
食材のラベルが全て0のとき\_\_background\_\_が1. 
	- db\_\*.h5:  
画像とラベルを含むcaffe用のデータベース. 画像は平均画像を引いたものなのでdata\_.npyとは異なる. 
caffeの仕様でhdf5形式のデータベースは, hdf5ファイルごとのシャッフル, および一つのhdf5ファイル内のデータのシャッフルはできるが, すべてまとめてシャッフルはできない. 
データの多様性確保と同じ画像が偏ってパラメータ更新に使われることがないように, 
suffixの番号がnのdb\_\*.h5には, 同じ番号nのdata\_.npyを反転させずに登録したものと, 番号(n-1)%(データベース数)のdata\_.npyを左右反転させて登録したものが含まれる.
	- train\_mean.npy:  
反転を含めたすべての画像の平均画像.
	- train\_dblist.txt:  
各行にdb\_\*.h5のパスを書いたもの. 
後でcaffeの設定ファイル内にこのテキストファイルのパスを書くことでh5ファイルを学習用データベースとして使う.

## 初期学習
### 前準備
- VGG-16のモデルの準備  
	- exp/caffe\_db\_test/0002/init/prototxtをコピーして読み込むデータベースとクラス数を書き直すのが一番楽だと思います.
	- https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
	- 初期値: ページ中ほどcaffemodel\_urlからVGG\_ILSVRC\_16\_layers.caffemodelをダウンロード(external/VGG16に保存)
	- モデル: VGG\_ILSVRC\_16\_layers\_deploy.prototxtをダウンロード(external/VGG16に保存)  
これをコピーして次のように変更. 以降-|で始まる行は削除, \+|で始まる行は追加を表す. 
		- exp/caffe\_db/(クラスタ番号)/init/prototxt/train.prototxt
			- 学習用
			- 入力をHDF5 Data Layerに設定
			- ここで読み込むデータベースも設定する.
```
-| input: "data"
-| input_dim: 10
-| input_dim: 3
-| input_dim: 224
-| input_dim: 224
+| layer {
+|   name: "data"
+|   type: "HDF5Data"
+|   top: "data"
+|   top: "labels"
+|   include {
+|     phase: TRAIN
+|   }
+|   hdf5_data_param {
+|     source: "path/to/exp/exp/caffe_db/(クラスタ番号)/init/train_dblist.txt"
+|     batch_size: 20 
+|     shuffle: true
+|   }
+| }
+| layer {
+|   name: "data"
+|   type: "HDF5Data"
+|   top: "data"
+|   top: "labels"
+|   include {
+|     phase: TEST 
+|   }
+|   hdf5_data_param {
+|     # validation用の正確なデータはなく損失を見てもどのくらい収束しているか判断できないと思ったので, 
+|     # 修論時にはphase:TESTにおいてtrainと同じファイルtrain_dblist.txtをsourceとして使っていた.
+|     source: "path/to/exp/exp/caffe_db/(クラスタ番号)/init/train_dblist.txt"
+|     batch_size: 1 
+|     shuffle: true
+|   }
+| }
```
			
- 一番下のINNER_PRODUCTとSOFTMAXを消して書き換え  

後のプログラムで呼び出せるように名前をfc8からcls\_scoreに(fc8だと初期値のcaffemodelとレイヤーの構造が一致せずエラー), 
最後の層はloss\_cls . 
num\_outputをクラスタに含まれる食材数+1(背景クラス)にする.  
マルチラベルに対応するためSigmoidCrossEntropyLossにする.  
```
-| layers {
-|   bottom: "fc7"
-|   top: "fc8"
-|   name: "fc8"
-|   type: INNER_PRODUCT
-|   inner_product_param {
-|     num_output: 1000
-|   }
-| }
-| layers {
-|   bottom: "fc8"
-|   top: "prob"
-|   name: "prob"
-|   type: SOFTMAX
-| }
+| layer {
+|   name: "cls_score"
+|   type: "InnerProduct"
+|   bottom: "fc7"
+|   top: "cls_score"
+|   param {
+|     lr_mult: 1
+|   }
+|   param {
+|     lr_mult: 2
+|   }
+|   inner_product_param {
+|     num_output: 3 #(クラスタに含まれる食材数+1)
+|     weight_filler {
+|       type: "gaussian"
+|       std: 0.01
+|     }
+|     bias_filler {
+|       type: "constant"
+|       value: 0
+|     }
+|   }
+| }
+| layer {
+|   name: "loss_cls"
+|   type: "SigmoidCrossEntropyLoss"
+|   bottom: "cls_score"
+|   bottom: "labels"
+|   top: "loss_cls"
+|   loss_weight: 1
+| }
```
		
- exp/caffe\_db/(クラスタ番号)/init/prototxt/deploy.prototxt
- input layerの設定(最初の次元を1にしないと1枚の入力に対して同じ値が10個返されたりする?)  

```
-| input: "data"
-| input_dim: 10
-| input_dim: 3
-| input_dim: 224
-| input_dim: 224
+| layer {
+|   name: "data"
+|   type: "Input"
+|   top: "data"
+|   input_param { shape: { dim: 1 dim: 3 dim: 224 dim: 224 } }
+| }
```
			
- train.prototxtと同様にfc8をcls_scoreに書き換え. 
最終層はSigmoidで名前はcls\_probにする.
```
-| layers {
-|   bottom: "fc7"
-|   top: "fc8"
-|   name: "fc8"
-|   type: INNER_PRODUCT
-|   inner_product_param {
-|     num_output: 1000
-|   }
-| }
-| layers {
-|   bottom: "fc8"
-|   top: "prob"
-|   name: "prob"
-|   type: SOFTMAX
-| }
+| layer {
+|   name: "cls_score"
+|   type: "InnerProduct"
+|   bottom: "fc7"
+|   top: "cls_score"
+|   param {
+|     lr_mult: 1
+|   }
+|   param {
+|     lr_mult: 2
+|   }
+|   inner_product_param {
+|     num_output: 3 #(クラスタに含まれる食材数+1)
+|     weight_filler {
+|       type: "gaussian"
+|       std: 0.01
+|     }
+|     bias_filler {
+|       type: "constant"
+|       value: 0
+|     }
+|   }
+| }
+| layer {
+|   name: "cls_prob"
+|   bottom: "cls_score"
+|   top: "cls_prob"
+|   type: "Sigmoid"
+| }
```
		
- exp/caffe\_db/(クラスタ番号)/init/prototxt/solver.prototxt
- 以下は例.  
exp/caffe\_db/(クラスタ番号)/init/img\_list\_\*.tsvの行数からデータ数を求める.  
1epochはデータ数\*2(左右反転)/20(バッチサイズ)で計算.  
testはあんまり意味が無いので適当.  
exp/caffe\_db/(クラスタ番号)/init/snapshotディレクトリをあらかじめ作っておく.  
```
net: "path/to/exp/caffe\_db/(クラスタ番号)/init/prototxt/train.prototxt"
snapshot_prefix: "path/to/exp/caffe\_db/(クラスタ番号)/init/snapshot/(prefix)"
base_lr: 0.001
lr_policy: "step"
stepsize: (1epoch) 
snapshot: (1epoch) 
average_loss: 1000 
max_iter: (15epoch) 
test_iter: 200 
test_interval: (1epoch) 
test_initialization: false
display: 40
gamma: 0.9
momentum: 0.9
weight_decay: 0.0005
solver_mode: GPU
```

### 実行ファイル
script/4\_training.sh
- (注) PREFIXはsolver.prototxtで設定したsnapshotのプレフィックス.  
ITERは特徴抽出に用いるモデルの学習回数.

### 出力
- exp/caffe\_db/(クラスタ番号)/init/snapshot/(prefix)  
caffeの出力結果. 
モデルのパラメータがcaffemodel. 
パラメータ含む途中結果がsolverstate(途中からやり直す場合に必要).
詳しくはcaffeのドキュメントとかを参照してください.
- exp/caffe\_db/(クラスタ番号)/refine/(ITER)  
特徴ベクトル抽出の結果を表す4つのファイル(.npy).
ファイルは全てインデックスでアクセスするnp.arrayであり, 
それぞのインデックスは同じ画像に対応している. 
Nは背景クラス以外のクラスのいずれかが1である画像数.
	- f\_imgs.npy  
切り出す前の画像パス,resize\_w,resize\_h,x,y,w,h(N\*7). 
img\_list.tsvに書かれているものと同じ. 
画像を一意に特定するために用いる.
	- f\_labels.npy  
ラベルベクトル(N\*(クラス数-1)). 
背景クラス以外のクラスのいずれかが1である画像のみ特徴ベクトルを抽出している.
	- f\_probs.npy  
各クラスの尤度(N\*クラス数).   
	- features.npy
特徴ベクトル(N\*4096). 


## 訓練データ選択 
### 前準備
- 評価用の訓練データのアノテーションを行うためのスクリプトを実行(script/annotation\_train.sh)
- cv2.imshowが動く環境で動かす(sshなどでは動かない)
- 画像と領域が表示される. 続いてコンソールに(食材名)?とグループに属する食材の数だけ表示されるので, 
それぞれ領域内に写っていればy, 写っていなければnを入力. 
- 出力は他のimg\_list\_\*.tsvと同じ形式. 列の最後のラベルの0と1が推定結果ではなくアノテーション結果になる.
- 出力はimg\_list\_*_annotation.tsv. 
形式はexp/caffe_db/(クラスタ番号)/init/img_list_*.tsvとほぼ同じで
最後ほうの列の__background__, 食材1,  食材2  …の0と1が部分画像に対するラベルの推定結果ではなく
アノテーション結果(部分画像に食材が写っていれば1, 写っていなければ0)になる.
- 出力先はこのREADMEでは../exp/caffe\_db/(クラスタ番号)/train\_annotationを想定. 
- 出力は随時保存される. 指定したoutput\_dirに同じファイルがあれば, Ctrl+Cで止めても次回続きから再開することが可能.
- 背景クラスが1のものは訓練データ選択の評価には必要ないので-bg\_skipをオプションにして省略

### 実行ファイル
script/5\_filtering.sh
- (注) ITERは特徴抽出に用いるモデルの学習回数.
- (注) Faster R-CNNは画像のパスと領域の情報さえあれば学習できるので, 
hdf5形式のデータベースを作る必要はない. 
よって訓練データ選択後はデータベースを作らず
Faster R-CNNの学習の入力として必要なexp/caffe\_db/(クラスタ番号)/refine/(ITER)/img\_list_*ur.tsvのみ出力
(各行に訓練データ選択後の部分画像の情報, すなわち元の画像と領域の情報を記述している).  
訓練データ選択をしてからVGG-16などFaster R-CNN以外のモデルで学習するなどの場合はデータベースを作り直す必要があり, 
その場合はtools/update\_label\_remove.pyではなくtools/refine\_db\_remove.py(その他のソースコードで説明)を用いる.

### 出力
- exp/caffe\_db/(クラスタ番号)/refine/(ITER)/fp\_estimation  
訓練データ選択結果. 
	- food(食材インデックス)/out_(食材インデックス).npy  
除去する画像のf\_imgs.npyにおけるインデックスのリスト. 
食材インデックスはexp/caffe\_db/(クラスタ番号)/init/img\_list\_\*.tsvの\_\_background\_\_以降の列に対応
(\_\_background\_\_が0)._  
	- noise.csv  
一行目が元の偽陽性訓練データの割合(修論ではp^-).
ニ行目が訓練データ選択後の偽陽性訓練データの割合(修論では\\hat{p}^-).
	- prob\_hist.pdf  
各クラスタに属する調理過程画像の尤度に対する積み上げヒストグラム.
修論では未使用. 
	- cluster2/prob\_hist.pdf cluster2/prob\_hist2.pdf  
訓練データの尤度に対する積み上げヒストグラム. 
TPが対象食材が写っており正しく真陽性訓練データと推定された画像, 
FNが対象食材が写っており誤って偽陽性訓練データと推定された画像, 
FPが対象食材が写っておらず誤って真陽性訓練データと推定された画像, 
TNが対象食材が写っておらず正しく偽陽性訓練データと推定された画像. 
prob\_hist2.pdfでTP+FNがTrue Positive, FP+TNがFalse Positive. 
修論ではprob\_hist.pdfは未使用, prob\_hist2.pdfは図9, 10, 11. 
	- cluster2/text\_flow.csv  
各行は順に上記のTP, FN, FP, TNの数.
各列は1列目がフローグラフを遡ることで対象食材のラベル(遡及ラベル)が付いた画像, 
2列目が対応する手順文章から対象食材のラベル(直接ラベル)が付いた画像の数. 
修論では1列目, 2列目の結果を足して, recall^-やrecall^+の計算に利用. 
		- 例
```
93,115
130,74
62,14
303,77
```
		- このときTP=93+115=208, FN=130+74=204, FP=62+14=86, TN=303+77=380
		- recall^- = TN / (TN + FP) =  0.815, recall^+ = TP / (TP + FN) = 0.505
- exp/caffe\_db/(クラスタ番号)/refine/(ITER)/img\_list\_\*ur.tsv  
ラベル除去後の訓練データ.
形式はexp/caffe\_db/(クラスタ番号)/init/img\_list\_\*.tsvと同じ. 


## Faster R-CNNによる学習 
### 前準備
- Faster R-CNNのモデルの準備  
	- exp/caffe\_db\_test/0002/refine/4199/frcnn/prototxtをコピーしてクラス数を書き直すのが一番楽だと思います.
	- https://github.com/rbgirshick/py-faster-rcnn
	- 上記URLのREADMEと同じ意味で$FRCNN\_ROOTを用いる
	- $FRCNN\_ROOT/py-faster-rcnn/models/pascal\_voc/VGG16/faster\_frcnn\_end2end/train.prototxtをコピーして以下のように変更
		- exp/caffe\_db/(クラスタ番号)/refine/(ITER)/frcnn/prototxt/train.prototxt  
			- クラス数を変更(num_classesやnum_outputもしくは21, 84で検索)
				- 11行目
```
-| param_str: "'num_classes': 21"
+| param_str: "'num_classes': 3" # グループに含まれる食材数+1
```
				- 530行目 roi-dataのparam\_str
```
-| param_str: "'num_classes': 21"
+| param_str: "'num_classes': 3" # グループに含まれる食材数+1
```
				- 620行目 cls_scoreのnum\_output
```
-| num_output: 21
+| num_output: 3 # グループに含まれる食材数+1
```
				- 643行目 bbox_predのnum\_output: **(グループに含まれる食材数+1)\*4なので注意** (矩形領域の座標なのでクラス数\*4)
```
-| num_output: 84 
+| num_output: 12 # (グループに含まれる食材数+1)*4
```
			- レイヤーをマルチラベル用に変更(元のfaster rcnnのコードではなくtools/frcnn以下の書き換えたコードを呼び出す)
				- 9行目 
```
-| module: 'roi_data_layer.layer'
+| module: 'frcnn.roi_data_layer.roi_data_layer'
```
				- 440行目
```
-| module: 'rpn.anchor_target_layer'
+| module: 'frcnn.anchor_target_layer'
```
				- 528行目
```
-| module: 'rpn.proposal_target_layer'
+| module: 'frcnn.proposal_target_layer'
```
			- 654行目 loss\_clsをSigmoidCrossEntropyLossに変える.  
```
-| layer {
-|   name: "loss_cls"
-|   type: "SoftmaxWithLoss"
-|   bottom: "cls_score"
-|   bottom: "labels"
-|   propagate_down: 1
-|   propagate_down: 0
-|   top: "loss_cls"
-|   loss_weight: 1
-| }
+| layer {
+|   name: "loss_cls"
+|   type: "SigmoidCrossEntropyLoss"
+|   bottom: "cls_score"
+|   bottom: "labels"
+|   top: "loss_cls"
+|   loss_weight: 1
+|   propagate_down: true  # backprop to prediction
+|   propagate_down: false # don't backprop to labels
+| }
```
	- $FRCNN\_ROOT/py-faster-rcnn/models/pascal\_voc/VGG16/faster\_frcnn\_end2end/test.prototxtをコピーして以下のように変更
		- exp/caffe\_db/(クラスタ番号)/refine/(ITER)/frcnn/prototxt/test.prototxt  
			- クラス数を変更(num_classesやnum_outputもしくは21, 84で検索)
				- 567行目 cls_scoreのnum\_output: (グループに含まれる食材数+1)  
```
-| num_output: 21
+| num_output: 3 # グループに含まれる食材数+1
```
				- 592行目 bbox_predのnum\_output: **(グループに含まれる食材数+1)\*4なので注意** (矩形領域の座標なのでクラス数\*4)
```
-| num_output: 84 
+| num_output: 12 # (グループに含まれる食材数+1)*4
```
			- 603行目 cls\_probをSigmoidに変える.  
```
-| layer {
-|   name: "cls_prob"
-|   type: "Softmax"
-|   bottom: "cls_score"
-|   top: "cls_prob"
-| }
+| layer {
+|   name: "cls_prob"
+|   bottom: "cls_score"
+|   top: "cls_prob"
+|   type: "Sigmoid"
+| }
```
	- exp/caffe\_db/(クラスタ番号)/refine/(ITER)/frcnn/prototxt/solver.prototxt  
以下は例.  
通常のcaffeの仕様と異なり, snapshotを保存するディレクトリはソースコード上で設定するので注意.
```
train_net: "path/to/exp/caffe\_db/(クラスタ番号)/refine/(ITER)/frcnn/prototxt/train.prototxt"
base_lr: 0.001
lr_policy: "step"
gamma: 0.1
stepsize: 10000
display: 20
average_loss: 1000
# iter_size: 1
momentum: 0.9
weight_decay: 0.0005
# We disable standard caffe solver snapshotting and implement our own snapshot
# function
snapshot: 0 #コード上で設定
# We still use the snapshot prefix, though
snapshot_prefix: "(prefix)" 
iter_size: 2
```
- tools/frcnn/\_init\_paths.pyのthis\_dirをインストールしたpy-faster-rcnnのディレクトリパスに変える.   

### 実行ファイル
script/6\_training\_frcnn.sh  
- (注)イテレーション回数の設定が必要(スクリプトに直接書き込み).
exp/caffe\_db/(クラスタ番号)/refine/(ITER)/img\_list\_\*.tsvの行数からデータ数を求める. 

###　出力 
- ../exp/caffe\_db/(クラスタ番号)/refine/(ITER)/frcnn/snapshot \
caffeの出力結果. 


## 評価 
### 前準備
- 評価用のテストデータのアノテーションを行うためのスクリプトを実行(script/annotation\_test.sh)
- cv2.imshowが動く環境で動かす(sshなどでは動かない)
- ほとんどscript/annotation\_train.shと同じで, こちらは背景クラスの画像にもアノテーションする. 
- 画像と領域が表示されるが**領域は無視**. 
続いてコンソールに(食材名)?とグループに属する食材の数だけ表示されるので, 
それぞれ画像に写っていればy, 写っていなければnを入力. 
- 出力先はこのREADMEでは../exp/caffe\_db/(クラスタ番号)/test\_annotationを想定. 

### 実行ファイル
script/7\_evaluation.sh
- (注)snapshotのプレフィックスとイテレーション回数の設定が必要(スクリプトに直接書き込み).

###　出力 
- exp/caffe\_db/(クラスタ番号)/refine/(ITER)/result.json  
初期学習のモデルによる予測結果.
各画像について結果の辞書型になっている.
boxesは[クラス数\*4]\*領域数のリストでクラス順にx,y,w,h.
predictsは[クラス数]\*領域数のリスト.
クラスの順番はexp/caffe\_db/(クラスタ番号)/init/img\_list\_\*.tsvの\_\_background\_\_以降の列に対応
(\_\_background\_\_が0)._ 
領域はSelectiveSearchで決めるのでどのクラスも同じx,y,w,hになっているが, Faster R-CNNの結果と合わせるために冗長に書いている.
- exp/caffe\_db/(クラスタ番号)/refine/(ITER)/frcnn/result.json  
Faster R-CNNのモデルによる予測結果.
形式は上に同じ.
- exp/caffe\_db/(クラスタ番号)/roc\_(食材インデックス).pdf  
スクリプトでは上記2つのroc曲線. 
コンソールには平均適合率(average precision)と偽陽性率0.2のときの真陽性率(require)も表示.
evaluation\_roc.pyのオプション
-result\_pathsに上のいずれかのresult.json, 
-result\_labelsに図の凡例, 
-stylesにmatplotlibの色と線の設定(例: k-(黒の実線), r--(赤の点線))を順に追加することで, 
一枚の画像に複数のroc曲線を表示できる.
- exp/caffe\_db/(クラスタ番号)/refine/(ITER)/result/food(食材インデックス), exp/caffe\_db/(クラスタ番号)/refine/(ITER)/frcnn/result/food(食材インデックス)  
結果の画像.
上位何件まで表示するかや色の設定はtools/module/Evaluation.pyの中身の引数を見てください.
	- O  
対象の食材が写っている調理過程画像の結果. 
デフォルトでは上位3件の領域で高い順にマゼンタ, 黄, 緑.
	- X  
対象の食材が写っていない調理過程画像の結果. 
デフォルトでは上位3件の領域で高い順にマゼンタ, 黄, 緑.
	- O\_req  
対象の食材が写っている調理過程画像の上位1件の領域を表示. 
偽陽性率が0.2未満になるまで上位から順に選択した時, 
選択されれば(対象の食材と判定されれば)緑の領域, 
選択されなければ(対象の食材ではないと判定されれば)赤の領域. 
	- X\_req  
対象の食材が写っていない調理過程画像の上位1件の領域を表示. 
偽陽性率が0.2未満になるまで上位から順に選択した時, 
選択されれば(対象の食材と判定されれば)緑の領域, 
選択されなければ(対象の食材ではないと判定されれば)赤の領域. 
	- raw  
元の調理過程画像
	- rect  
領域を全て書いた画像


## 訓練データ選択の評価
### 前準備
tools/learning/extract\_features.py(script/4\_training.sh)によって, 
特徴量抽出を評価したいエポック数分だけ行い, 
exp/caffe\_db/(クラスタ番号)/refine/(ITER)に保存しておく. 

### 実行ファイル
script/5.5\_evaluate\_filtering.sh  
- (注)1エポックのイテレーション回数(EPOCH), 
何エポック目まで評価するか(N\_EPOCH), 
元の偽陽性訓練データの割合(修論ではp^-:FIRST\_RATE),
出力する食材のインデックスが必要. 

### 出力
- exp/caffe\_db/(クラスタ番号)/fp\_estimation/fp\_estimation\_\*  
各イテレーションにおけるscript/5\_filtering.shの出力と同じ
- exp/caffe\_db/(クラスタ番号)/fp\_estimation/result.csv  
各列は順に
	- エポック数
	- 食材1の訓練データ選択後の偽陽性訓練データの割合(修論の\\hat{p_}^-)
	- 食材1の訓練データ選択後の真陽性訓練データの再現率(修論のrecall^+)
	- 食材1の訓練データ選択後の遡及ラベルに関する真陽性訓練データの再現率(修論では未使用)
	- 食材1の訓練データ選択後の直接ラベルに関する真陽性訓練データの再現率(修論では未使用)
	- 食材2の訓練データ選択後の偽陽性訓練データの割合(修論の\\hat{p_}^-)
	- (以降各食材に関して同じように続く)...
- exp/caffe\_db/(クラスタ番号)/fp\_estimation/fp\_rate.pdf  
指定した食材の各エポック数における偽陽性訓練データの割合を図示(修論では図7, 図14).


#その他ソースコード
- 訓練データ選択後, VGG-16のモデルで学習
	- tools/learning/refine\_db\_remove.py
		- 訓練データ選択後データベースを作り直す.  
		- 引数はtools/learning/update\_label\_remove.pyと同じ. 
		- 出力はtools/learning/generate\_db.pyと同じ.
- 直接ラベルのみで学習するためのデータベースを生成
	- tools/learning/generate\_db\_step.py  
		- tools/learning/generate\_db.py(script/3\_generate\_db.sh)で生成したデータベースに対して直接ラベル(フローグラフを遡らずに付けたラベル)のデータベースを作るためのプログラム
		- 比較用なので比較対象(遡及ラベル)のimg\_listを先に作っていることが前提. 
	もし一から作るならtools/learning/generate\_db.py中のimg\_list.add\_labelの引数にstp=Trueを追加すれば同じものが作れる.
		- 引数は-output\_dirを任意のディレクトリ, -db\_dirをtools/learning/generate\_db.pyのoutput\_dirに設定したディレクトリに設定. 
	他はtools/learning/generate\_db.pyと同じ. 
		- train\_mean.npyとdata\_\*.npyはもとのdb\_dirにあるものと同じなので必要ならこれを使う.  
- フローグラフを可視化 
	- tools/dot

#注意事項
- ソースコードの中に何故かsys.path.append(os.path.join(\_\_file\_\_, "../"))で修論時問題なく動いているものがありました. 
tools/moduleがimportエラーなどの場合, 正しくはsys.path.append(os.path.join(os.path.dirname(\_\_file\_\_), "../"))なので書き換えてください.
