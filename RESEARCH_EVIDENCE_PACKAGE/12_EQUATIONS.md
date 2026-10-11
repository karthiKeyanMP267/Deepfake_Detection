# Mathematical Formulations & Equation Derivations

## 1. Feature Aggregation Formulations

### A. Vision Feature Aggregation
Let $\mathbf{f}_{	ext{eff}} \in \mathbb{R}^{1280}$, $\mathbf{f}_{	ext{swin}} \in \mathbb{R}^{768}$, and $\mathbf{f}_{	ext{mouth}} \in \mathbb{R}^{512}$ be the feature embeddings extracted by the three visual foundation encoders. The concatenated visual representation is:
$$\mathbf{f}_{v,	ext{concat}} = [\mathbf{f}_{	ext{eff}} \,\|\, \mathbf{f}_{	ext{swin}} \,\|\, \mathbf{f}_{	ext{mouth}}] \in \mathbb{R}^{2560}$$
The unified visual embedding $\mathbf{f}_v \in \mathbb{R}^{1280}$ is obtained via linear projection:
$$\mathbf{f}_v = \mathbf{W}_{	ext{agg},v} \mathbf{f}_{v,	ext{concat}} + \mathbf{b}_{	ext{agg},v}$$
where $\mathbf{W}_{	ext{agg},v} \in \mathbb{R}^{1280 	imes 2560}$ and $\mathbf{b}_{	ext{agg},v} \in \mathbb{R}^{1280}$.
*(Source: `models/vision_feature_aggregation.py:85-88`)*

### B. Audio Feature Aggregation
Let $\mathbf{f}_{	ext{whisper}} \in \mathbb{R}^{512}$, $\mathbf{f}_{	ext{wav2vec2}} \in \mathbb{R}^{768}$, and $\mathbf{f}_{	ext{spec}} \in \mathbb{R}^{256}$ be the acoustic feature embeddings. The concatenated acoustic representation is:
$$\mathbf{f}_{a,	ext{concat}} = [\mathbf{f}_{	ext{whisper}} \,\|\, \mathbf{f}_{	ext{wav2vec2}} \,\|\, \mathbf{f}_{	ext{spec}}] \in \mathbb{R}^{1536}$$
The unified acoustic embedding $\mathbf{f}_a \in \mathbb{R}^{512}$ is obtained via linear projection:
$$\mathbf{f}_a = \mathbf{W}_{	ext{agg},a} \mathbf{f}_{a,	ext{concat}} + \mathbf{b}_{	ext{agg},a}$$
where $\mathbf{W}_{	ext{agg},a} \in \mathbb{R}^{512 	imes 1536}$ and $\mathbf{b}_{	ext{agg},a} \in \mathbb{R}^{512}$.
*(Source: `models/audio_feature_aggregation.py:85-88`)*

---

## 2. Projection Heads & Metric Space Normalization

The unified features $\mathbf{f}_v$ and $\mathbf{f}_a$ are mapped into a shared metric space $\mathbb{R}^{256}$ via multi-layer perceptron (MLP) projection heads with batch normalization, dropout, and $L_2$ spherical normalization:

### Visual Projection Head:
$$\mathbf{h}_v = 	ext{ReLU}(	ext{BatchNorm1d}(\mathbf{W}_{v1} \mathbf{f}_v + \mathbf{b}_{v1})) \in \mathbb{R}^{512}$$
$$	ilde{\mathbf{e}}_v = \mathbf{W}_{v2} 	ext{Dropout}_{0.2}(\mathbf{h}_v) + \mathbf{b}_{v2} \in \mathbb{R}^{256}$$
$$\mathbf{e}_v = rac{	ilde{\mathbf{e}}_v}{\|	ilde{\mathbf{e}}_v\|_2} \in \mathbb{R}^{256} \quad 	ext{such that } \|\mathbf{e}_v\|_2 = 1.0$$
*(Source: `models/projection_heads.py:44-61`)*

### Acoustic Projection Head:
$$\mathbf{h}_a = 	ext{ReLU}(	ext{BatchNorm1d}(\mathbf{W}_{a1} \mathbf{f}_a + \mathbf{b}_{a1})) \in \mathbb{R}^{512}$$
$$	ilde{\mathbf{e}}_a = \mathbf{W}_{a2} 	ext{Dropout}_{0.2}(\mathbf{h}_a) + \mathbf{b}_{a2} \in \mathbb{R}^{256}$$
$$\mathbf{e}_a = rac{	ilde{\mathbf{e}}_a}{\|	ilde{\mathbf{e}}_a\|_2} \in \mathbb{R}^{256} \quad 	ext{such that } \|\mathbf{e}_a\|_2 = 1.0$$
*(Source: `models/projection_heads.py:44-61`)*

---

## 3. Audio-Visual Cosine Consistency Metric

Because $\mathbf{e}_v$ and $\mathbf{e}_a$ are unit-normalized vectors ($\|\mathbf{e}_v\|_2 = \|\mathbf{e}_a\|_2 = 1$), their cosine similarity reduces directly to their inner product:
$$S_{AV} = \cos(\mathbf{e}_v, \mathbf{e}_a) = rac{\mathbf{e}_v \cdot \mathbf{e}_a}{\|\mathbf{e}_v\|_2 \|\mathbf{e}_a\|_2} = \mathbf{e}_v^T \mathbf{e}_a = \sum_{k=1}^{256} e_{v,k} e_{a,k} \in [-1.0, +1.0]$$
*(Source: `models/contrastive_module.py:72-74` and `agent/reasoning.py:115-116`)*

---

## 4. Cross-Modal InfoNCE Contrastive Loss

Given a minibatch of $B$ video-audio embedding pairs $\{(\mathbf{e}_{v,i}, \mathbf{e}_{a,i})\}_{i=1}^B$ with temperature $	au = 0.07$:

$$\mathcal{L}_{i2a} = -rac{1}{B} \sum_{i=1}^B \log rac{\exp\left(rac{\mathbf{e}_{v,i}^T \mathbf{e}_{a,i}}{	au}ight)}{\sum_{j=1}^B \exp\left(rac{\mathbf{e}_{v,i}^T \mathbf{e}_{a,j}}{	au}ight)}$$

$$\mathcal{L}_{a2i} = -rac{1}{B} \sum_{i=1}^B \log rac{\exp\left(rac{\mathbf{e}_{a,i}^T \mathbf{e}_{v,i}}{	au}ight)}{\sum_{j=1}^B \exp\left(rac{\mathbf{e}_{a,i}^T \mathbf{e}_{v,j}}{	au}ight)}$$

$$\mathcal{L}_{	ext{contrastive}} = rac{\mathcal{L}_{i2a} + \mathcal{L}_{a2i}}{2}$$
*(Source: `models/contrastive_module.py:76-96`)*

---

## 5. Adaptive Multimodal Attention Fusion

The joint multimodal embedding $\mathbf{u} = [\mathbf{e}_v \,\|\, \mathbf{e}_a] \in \mathbb{R}^{512}$ is passed through an attention network to compute dynamic modality weights:
$$\mathbf{z} = \mathbf{W}_2 	ext{ReLU}(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^2$$
where $\mathbf{W}_1 \in \mathbb{R}^{128 	imes 512}$, $\mathbf{b}_1 \in \mathbb{R}^{128}$, $\mathbf{W}_2 \in \mathbb{R}^{2 	imes 128}$, $\mathbf{b}_2 \in \mathbb{R}^2$.
Modality attention weights $[lpha_v, lpha_a]$ are obtained via softmax normalization:
$$lpha_v = rac{\exp(z_1)}{\exp(z_1) + \exp(z_2)}, \quad lpha_a = rac{\exp(z_2)}{\exp(z_1) + \exp(z_2)} \quad (lpha_v + lpha_a = 1.0)$$
The fused embedding $\mathbf{e}_{	ext{fused}} \in \mathbb{R}^{512}$ is computed as:
$$\mathbf{e}_{	ext{fused}} = [lpha_v \mathbf{e}_v \,\|\, lpha_a \mathbf{e}_a] \in \mathbb{R}^{512}$$
**Missing Audio Fallback Protocol**:
$$\mathbf{e}_{	ext{fused}} = egin{cases} [lpha_v \mathbf{e}_v \,\|\, lpha_a \mathbf{e}_a] & 	ext{if } 	ext{audio\_available} = 	ext{True} \ [\mathbf{e}_v \,\|\, \mathbf{0}] & 	ext{if } 	ext{audio\_available} = 	ext{False} \end{cases}$$
*(Source: `models/adaptive_fusion.py:77-133`)*

---

## 6. Deepfake Classification Head & Objective Function

### Classification Head:
$$\mathbf{h}_1 = 	ext{Dropout}_{0.3}(	ext{ReLU}(	ext{BatchNorm1d}(\mathbf{W}_{c1} \mathbf{e}_{	ext{fused}} + \mathbf{b}_{c1}))) \in \mathbb{R}^{256}$$
$$\mathbf{h}_2 = 	ext{Dropout}_{0.3}(	ext{ReLU}(	ext{BatchNorm1d}(\mathbf{W}_{c2} \mathbf{h}_1 + \mathbf{b}_{c2}))) \in \mathbb{R}^{128}$$
$$\mathbf{z}_{	ext{out}} = \mathbf{W}_{c3} \mathbf{h}_2 + \mathbf{b}_{c3} \in \mathbb{R}^2$$
$$P(y = k \mid \mathbf{x}) = rac{\exp(z_{	ext{out},k})}{\exp(z_{	ext{out},0}) + \exp(z_{	ext{out},1})}, \quad k \in \{0 (	ext{Real}), 1 (	ext{Fake})\}$$
*(Source: `models/classifier.py:36-71`)*

### Focal Loss Objective:
$$	ext{FL}(p_t) = -lpha_t (1 - p_t)^\gamma \log(p_t)$$
where $\gamma = 2.0$, $lpha_t = 0.25$ for $y=1$ (Fake) and $lpha_t = 0.75$ for $y=0$ (Real).
*(Source: `training/losses.py:6-34`)*

### Total Composite Loss:
$$\mathcal{L}_{	ext{total}} = 1.0 \cdot 	ext{FL}(p_t) + 0.2 \cdot \mathcal{L}_{	ext{contrastive}}$$
*(Source: `training/losses.py:80-87`)*
