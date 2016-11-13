import numpy as np
from scipy.special import expit
import sys
from sklearn.base import BaseEstimator


class MLPGradientCheck(BaseEstimator):
    
    
    #NeauralNetMLPの初期化
    #l1とl2は正則化パラメータλ
    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001,
                            alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1 #正則化パラメータλの大きさ
        self.l2 = l2 #正則化パラメータλの大きさ
        self.epochs = epochs
        self.eta = eta #学習率
        self.alpha = alpha #モーメンタム学習の一つ手前の勾配係数(偏微分係数)
        self.decrease_const = decrease_const #適応学習率の減少定数
        self.shuffle = shuffle # データのシャッフル
        self.minibatches = minibatches #各エポックでのミニバッチ数
        
    #ラベルのエンコード
    #出力層×サンプル数の行列を作って
    #各列がサンプルを表しone of k になるようにする(ダミー変数のようなもの)
    def _encode_labels(self, y, k):
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot
    
    #重みの初期化
    def _initialize_weights(self):
        #-1から1の間の一様分布からランダムに抽出
        #これは隠れ層
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features +1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        #ここから出力層
        w2 = np.random.uniform(-1.0,1.0, size=self.n_output*(self.n_hidden +1))
        w2 = w2.reshape(self.n_output, self.n_hidden+1)
        return w1, w2
    
    #シグモイド関数
    def _sigmoid(self, z):
        return expit(z) #expitは1.0/(1.0 + np.exp(-z))に等しい
    
    #シグモイド関数の勾配(偏微分係数)要するに1.0/(1.0 + np.exp(-z))を微分したもの
    def _sigmoid_gradient(self, z):
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)
    
    #バイアスユニットの追加(各入力特徴量の一番目)
    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new
    
    # フィードフォワード
    def _feedforward(self, X, w1, w2):
        #一層目
        a1 = self._add_bias_unit(X, how='column')
        #二層目
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        #三層目
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3
    
    #L2ペナルティ項の係数
    #lambdaは正則化パラメータ
    def _L2_reg(self, lambda_, w1, w2):
        #1列目はバイアスユニットなので正則化しない
        return (lambda_/2.0) * (np.sum(w1[:, 1:]**2) + np.sum(w2[:, 1:]**2))
    
    #L1ペナルティ項の係数
    def _L1_reg(self, lambda_, w1, w2):
        #1列目はバイアスユニットなので正則化しない
        return (lambda_/2.0)*(np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())
    
    #ロジスティック関数
    #コストを得る
    #y_encは正解データをy_encに変換したもの
    #outputはa3
    def _get_cost(self, y_enc, output, w1,w2):
        #最初のcostはロジスティック関数のコスト関数(対数尤度関数)p58
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        #バックプロバゲーション
        #シグマは各層のコスト関数を総入力zで微分したもの(誤差項)※ここでは正則化項は関係ない
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        #gradはコスト関数の偏微分係数(コスト関数をWで微分したもの)
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        #正則化
        grad1[:, 1:] += (w1[:, 1:]*(self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:]*(self.l1 + self.l2))
        return grad1, grad2
    
    #フィードフォワードにによる予測
    def predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        #np.argmaxは列ごとに一番大きな値のインデックスを返す
        y_pred = np.argmax(z3, axis=0)
        return y_pred
    
    #モデルをデータに適合させる
    def fit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output) #yをonehotエンコーディング
        delta_w1_prev = np.zeros(self.w1.shape) #一つ前の重みを初期化
        delta_w2_prev = np.zeros(self.w2.shape) #一つ前の重みを初期化
        for i in range(self.epochs):
            #適応学習率
            #適応学習率の減少定数を加味している
            self.eta /= (1+self.decrease_const*i)
            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                #数字がフラッシュする
                sys.stderr.flush()
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                #サンプルの順番入れ替えX_dataは行、y_encは列がサンプルになっていることに注意
                X_data, y_enc = X_data[idx], y_enc[:, idx]
            #array_split(配列, 何個に分けるか)
            #y_data.shape[0]はサンプル数
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                #フィードフォワード
                #一回目はまだw1,w2は一様分布から抽出した乱数
                a1, z2, a2, z3, a3 = self._feedforward(X_data[idx], self.w1, self.w2)
                #コストを得る
                #出力層の対数尤度関数*-1とw1,w2の両方のL1正則化とL2正則化を足したもの
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3, w1 = self.w1, w2 = self.w2)
                self.cost_.append(cost)
                #バックプロバゲーションによる勾配の計算
                #重みを更新するのに必要な変化率を求める
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:,idx],
                                                 w1 = self.w1, w2 = self.w2)
                
                ##勾配チェックの始まり(gradがうまく計算できているか)
                grad_diff = self._gradient_checking(X=X_data[idx], y_enc=y_enc[:, idx],
                                                   w1= self.w1, w2= self.w2, epsilon=1e-5, grad1=grad1, grad2=grad2)
                
                if grad_diff<= 1e-7:
                    print('Ok: %s' % grad_diff)
                elif grad_diff<= 1e-4:
                    print('Warning: %s' % grad_diff)
                else :
                    print('Problem: %s' % grad_diff)
                    
                ##勾配チェック終了
                
                #重みの更新
                #delta_w1, delta_w2は入力層、中間層でのコスト関数の微分(ゼロにしたいもの且つ傾きであることに注意)に学習率をかけたもの
                #傾きが+ならば-方向に重みを動かすことで大局的最小に向かう
                delta_w1, delta_w2 = self.eta * grad1, self.eta*grad2
                self.w1 -= (delta_w1 + (self.alpha*delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha*delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
                
        return self
    
        def _gradient_checking(self, X, y_enc, w1, w2, epsilon, grad1, grad2):
        """ 勾配チェックの適用(デバッグ専用)
         
        戻り値
        ---------------
        relative_error : float
            数値的に近似された勾配とバックプロパゲーションによる勾配の間の相対誤差
        
        """
            
            
        #入力層を隠れ層に結合する重み行列w1から数値勾配を算出
        num_grad1 = np.zeros(np.shape(w1))
        epsilon_ary1 = np.zeros(np.shape(w1))
        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                #w1と同じ大きさですべてイプシロンの行列を作る
                epsilon_ary1[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1 - epsilon_ary1, w2)
                cost1 = self._get_cost(y_enc, a3,  w1 - epsilon_ary1, w2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1 + epsilon_ary1, w2)
                cost2 = self._get_cost(y_enc, a3,  w1 + epsilon_ary1, w2)
                num_grad1[i, j] = (cost2 - cost1) / (2*epsilon)
                epsilon_ary1[i, j] = 0
        
                
        #入力層を隠れ層に結合する重み行列w2から数値勾配を算出
        num_grad2 = np.zeros(np.shape(w2))
        epsilon_ary2 = np.zeros(np.shape(w2))
        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                #w1と同じ大きさですべてイプシロンの行列を作る
                epsilon_ary2[i, j] = epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 - epsilon_ary2)
                cost1 = self._get_cost(y_enc, a3,  w1, w2 - epsilon_ary2)
                a1, z2, a2, z3, a3 = self._feedforward(X, w1, w2 + epsilon_ary2)
                cost2 = self._get_cost(y_enc, a3,  w1, w2 + epsilon_ary2)
                num_grad2[i, j] = (cost2 - cost1) / (2*epsilon)
                epsilon_ary2[i, j] = 0
                    
        #数値勾配と解析的勾配の列ベクトル化
        num_grad = np.hstack((num_grad1.flatten(), num_grad2.flatten()))
        grad = np.hstack((grad1.flatten(), grad2.flatten()))
        #数値勾配と解析的勾配の差のノルム
        norm1= np.linalg.norm(num_grad - grad)
        #数値勾配のノルム
        norm2= np.linalg.norm(num_grad)
        #解析的勾配のノルム
        norm3= np.linalg.norm(grad)
        #相対誤差を計算
        relative_error = norm1 / (norm2 + norm3)
        return relative_error 
    