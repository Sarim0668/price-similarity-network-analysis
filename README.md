# 📊 Price Similarity Network Analysis of Pakistani Cities

## A Discrete Mathematical Exploration of Economic Connections

### 👥 Team Members

| Name | ID | Section |
|------|-----|---------|
| Muhammad Sarim | 24I-0668 | A |
| Abdul Moeed | 24I-0720 | A |
| Muhammad Zulqarnain | 24I-0729 | A |

### 🎥 Project Demo

[![Project Video](https://img.youtube.com/vi/fuzGm9K0Q14/0.jpg)](https://youtu.be/fuzGm9K0Q14?si=e_-1ApDWw9dH7Lyo)

---

## 📖 Overview

This project presents a comprehensive analysis of economic relationships between cities in Pakistan using **Consumer Price Index (CPI)** data. By applying discrete mathematical concepts including graph theory, relations, partial orders, and centrality measures, we uncover hidden economic connections in urban centers and identify cities that play crucial roles in the national price network.

---

## 🧮 Mathematical Framework

### 1. Network Construction

For each category *c* and year *y*, a relation *R* on cities is defined:
(C_i, C_j) ∈ R_y,c ⇔ CosineSim_y,c(C_i, C_j) ≥ τ


### 2. Cosine Similarity
CosineSim(C_i, C_j) = (p_i · p_j) / (||p_i|| × ||p_j||)


### 3. Temporal Partial Order
G_y1 ⊆ G_y2 ⇔ E_y1 ⊆ E_y2


### 4. Weighted Influence Score
S_i,y = Σ_c α_c [ Σ_k w_k Centrality_k(i) ]


---

## 📊 Data Structure

### Categories & Economic Weights

| Category | Weight |
|----------|--------|
| Food Staples & Grains | 25% |
| Meat, Poultry & Dairy | 15% |
| Oils, Condiments & Sweeteners | 10% |
| Fruits & Vegetables | 10% |
| Non-Food Essentials | 15% |
| Utilities & Transport | 20% |
| Clothing & Miscellaneous | 5% |

### Cities Analyzed
Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad, Multan, Peshawar, Quetta, Gujranwala, Sialkot, and more.

---

## 🛠️ Technical Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Graph Analysis | NetworkX |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (cosine similarity) |


## 🏆 Key Findings
Top 5 Cities (Equal Weighting)
Rank	City	Score	Key Characteristic
1	Karachi	0.845	Pakistan's economic hub
2	Lahore	0.812	Strong eigenvector centrality
3	Islamabad	0.789	Balanced across measures
4	Faisalabad	0.756	Industrial center
5	Rawalpindi	0.732	Efficient price transmission
Top 5 Cities (Correlation-Based Weighting)
Rank	City	Score	Key Characteristic
1	Karachi	0.831	Diverse centrality strength
2	Quetta	0.798	Critical bridging role ↑
3	Lahore	0.785	Slight adjustment
4	Peshawar	0.762	Distinct closeness centrality ↑
5	Islamabad	0.741	Rebalanced contributions


## 📈 Key Visualizations
Similarity Heatmap (Food Staples - Year 2022)
Shows price pattern alignment between cities

Thresholded Network (τ = 0.7)
Cities as nodes, edges represent strong price correlations

Temporal Hasse Diagram
Captures year-to-year network evolution

City Ranking Comparison
Effect of different weighting schemes on city importance

## 🔬 Technical Insights
### 1. Threshold Analysis
τ = 0.6: Dense networks, broad similarity patterns

τ = 0.7: Sparse but meaningful economic connections

τ = 0.8: Fragmenting networks, strongest relationships only

### 2. Weighting Scheme Effects
Equal Weighting: Balanced view, consistent performers

Correlation-Based: Highlights specialized roles (bridges, hubs)

Entropy-Based: Prioritizes distributional uniqueness

### 3. Mathematical Interpretation
Quetta's rise: High betweenness centrality provides unique information

Peshawar's emergence: Distinct closeness centrality pattern

Faisalabad's drop: Correlated measures reduce unique contribution



---

## 🚀 How to Run

### Clone Repository
```bash
git clone https://github.com/your-username/price-similarity-network-analysis.git
cd price-similarity-network-analysis

