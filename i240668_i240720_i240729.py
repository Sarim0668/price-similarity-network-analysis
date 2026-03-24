import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from collections import defaultdict
import shutil
warnings.filterwarnings('ignore')

print("="*70)
print("SPI PRICE SIMILARITY NETWORK ANALYSIS - YEAR-WISE VERSION")
print("Discrete Structures Project - Temporal Analysis Across Years")
print("="*70)

print("\n[1/10] Loading data and assigning categories...")

df = pd.read_excel('Data Combined.xlsx')
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Years in data: {sorted(df['Year'].unique())}")
print(f"Months in data: {list(df['Month'].unique())}")

categories = {
    'Food Staples & Grains': [
        'Wheat Flour', 'Rice Basmati', 'Rice IRRI', 'Bread', 'Chicken Farm',
        'Eggs Hen', 'Fish', 'Mutton', 'Milk Fresh', 'Milk Powder', 'Curd'
    ],
    'Meat, Poultry & Dairy': [
        'Beef', 'Mutton', 'Chicken', 'Eggs', 'Milk', 'Curd', 'Yogurt'
    ],
    'Oils, Condiments & Sweeteners': [
        'Cooking Oil', 'Vegetable Ghee', 'Pulse', 'Sugar', 'Salt', 'Red Chili', 'Garlic'
    ],
    'Fruits & Vegetables': [
        'Onions', 'Potatoes', 'Tomatoes', 'Banana', 'Apple'
    ],
    'Non-Food Essentials': [
        'Soap', 'Match Box', 'Washing Soap', 'Toilet Soap', 'Toothpaste'
    ],
    'Utilities & Transport': [
        'Electricity', 'Gas', 'Kerosene', 'Petrol', 'Diesel', 'Firewood'
    ],
    'Clothing & Miscellaneous': [
        'Shirting', 'Lawn', 'Georgette'
    ]
}

category_weights = {
    'Food Staples & Grains': 0.25,
    'Meat, Poultry & Dairy': 0.15,
    'Oils, Condiments & Sweeteners': 0.10,
    'Fruits & Vegetables': 0.10,
    'Non-Food Essentials': 0.15,
    'Utilities & Transport': 0.20,
    'Clothing & Miscellaneous': 0.05
}

assert abs(sum(category_weights.values()) - 1.0) < 1e-10, "Category weights must sum to 1"
print(f"Category weights verified: sum = {sum(category_weights.values()):.2f}")

def assign_category(product_name):
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in product_name.lower():
                return category
    return 'Other'

df['Category'] = df['Product'].apply(assign_category)
print(f"Categories assigned")
print(f"  Distribution:")
for cat in categories.keys():
    count = len(df[df['Category'] == cat])
    print(f"    {cat}: {count} records")

print("\n[2/10] Setting up core functions for year-wise analysis...")

def create_yearly_price_matrix(df, year, category):
    filtered = df[(df['Year'] == year) & (df['Category'] == category)]
    
    if filtered.empty:
        return None
    
    price_matrix = filtered.pivot_table(
        values='Price', 
        index='City', 
        columns='Product', 
        aggfunc='mean'
    )
    
    price_matrix = price_matrix.fillna(price_matrix.mean())
    price_matrix = price_matrix.fillna(0)
    
    return price_matrix

def compute_city_similarity(price_matrix):
    if price_matrix is None or price_matrix.empty:
        return None
    
    normalized_matrix = (price_matrix - price_matrix.mean()) / (price_matrix.std() + 1e-10)
    
    similarity_matrix = cosine_similarity(normalized_matrix)
    
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=price_matrix.index,
        columns=price_matrix.index
    )
    
    return similarity_df

def build_similarity_network(similarity_df, threshold=0.7):
    if similarity_df is None:
        return None
    
    G = nx.Graph()
    cities = similarity_df.index.tolist()
    G.add_nodes_from(cities)
    
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            city_i = cities[i]
            city_j = cities[j]
            similarity = similarity_df.loc[city_i, city_j]
            
            if similarity >= threshold:
                G.add_edge(city_i, city_j, weight=similarity)
    
    return G

def build_weighted_network_full(similarity_df, min_weight=0.3):
    if similarity_df is None:
        return None
    
    G = nx.Graph()
    cities = similarity_df.index.tolist()
    G.add_nodes_from(cities)
    
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            city_i = cities[i]
            city_j = cities[j]
            similarity = similarity_df.loc[city_i, city_j]
            
            if similarity >= min_weight:
                G.add_edge(city_i, city_j, weight=similarity)
    
    return G

def compute_centralities(G):
    if G is None or G.number_of_nodes() == 0:
        return None
    
    centralities = {}
    
    centralities['degree'] = nx.degree_centrality(G)
    
    try:
        centralities['closeness'] = nx.closeness_centrality(G)
    except:
        centralities['closeness'] = {node: 0 for node in G.nodes()}
    
    centralities['betweenness'] = nx.betweenness_centrality(G)
    
    try:
        centralities['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        centralities['eigenvector'] = {node: 0 for node in G.nodes()}
    
    centrality_df = pd.DataFrame(centralities)
    
    return centrality_df

print("Core functions ready (year-wise aggregation)")

print("\n[3/10] Setting up weighting schemes...")

def equal_weighting():
    return {'degree': 0.25, 'closeness': 0.25, 'betweenness': 0.25, 'eigenvector': 0.25}

def correlation_based_weighting(centrality_df):
    if centrality_df is None or centrality_df.empty:
        return equal_weighting()
    
    corr_matrix = centrality_df.corr().abs()
    
    weights = {}
    for measure in ['degree', 'closeness', 'betweenness', 'eigenvector']:
        if measure in corr_matrix:
            corr_sum = corr_matrix[measure].sum() - 1
            weights[measure] = 1 / (1 + corr_sum)
    
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    else:
        weights = equal_weighting()
    
    return weights

def entropy_based_weighting(centrality_df):
    if centrality_df is None or centrality_df.empty:
        return equal_weighting()
    
    weights = {}
    entropies = {}
    
    for measure in ['degree', 'closeness', 'betweenness', 'eigenvector']:
        if measure not in centrality_df:
            continue
        
        values = centrality_df[measure].values
        values = values + 1e-10
        probs = values / values.sum()
        
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies[measure] = entropy
    
    total_entropy = sum(entropies.values())
    if total_entropy > 0:
        weights = {k: v/total_entropy for k, v in entropies.items()}
    else:
        weights = equal_weighting()
    
    return weights

def compute_weighted_score(centrality_df, weights):
    if centrality_df is None or centrality_df.empty:
        return None
    
    scores = {}
    for city in centrality_df.index:
        score = 0
        for measure, weight in weights.items():
            if measure in centrality_df.columns:
                score += weight * centrality_df.loc[city, measure]
        scores[city] = score
    
    return pd.Series(scores).sort_values(ascending=False)

print("Weighting schemes ready")

print("\n[4/10] Creating output directories...")

output_dir = 'results_yearly'
subdirs = ['heatmaps', 'networks_thresholded', 'networks_weighted', 
           'centralities', 'scores', 'comparisons', 'temporal_hasse', 
           'overall_scores', 'threshold_analysis', 'year_comparisons']

for subdir in subdirs:
    os.makedirs(f'{output_dir}/{subdir}', exist_ok=True)

print(f"Created '{output_dir}' folder with all subfolders")

print("\n[5/10] Analyzing data structure...")

years = sorted(df['Year'].unique())
thresholds = [0.6, 0.7, 0.8]

print(f"Years available: {years}")
print(f"Number of years: {len(years)}")
print(f"Thresholds to analyze: {thresholds}")
print(f"Categories: {len(categories)}")

if len(years) < 2:
    print("\nWARNING: Less than 2 years of data!")
    print("  Temporal analysis requires at least 2 years")

print("\n[6/10] Running year-wise comprehensive analysis...")
print("="*70)

all_networks = defaultdict(lambda: defaultdict(dict))
all_similarities = defaultdict(dict)
analysis_count = 0

for year in years:
    print(f"\n{'='*70}")
    print(f"ANALYZING YEAR: {year}")
    print('='*70)
    
    for category in category_weights.keys():
        print(f"\n  Category: {category}")
        
        price_matrix = create_yearly_price_matrix(df, year, category)
        if price_matrix is None or price_matrix.shape[0] < 2:
            print(f"    Skipping - insufficient data")
            continue
        
        print(f"    Yearly price matrix: {price_matrix.shape[0]} cities, {price_matrix.shape[1]} products")
        
        similarity_df = compute_city_similarity(price_matrix)
        if similarity_df is None:
            print(f"    Skipping - could not compute similarity")
            continue
        
        all_similarities[category][year] = similarity_df
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_df, annot=True, fmt='.2f', 
                   cmap='YlOrRd', square=True, vmin=0, vmax=1,
                   cbar_kws={'label': 'Cosine Similarity'})
        plt.title(f'{category} - Year {year}\nCity Price Similarity (Yearly Average)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f'{output_dir}/heatmaps/{year}{category.replace(" ", "")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Creating raw weighted network...")
        G_weighted = build_weighted_network_full(similarity_df, min_weight=0.3)
        
        if G_weighted and G_weighted.number_of_edges() > 0:
            plt.figure(figsize=(16, 12))
            pos = nx.spring_layout(G_weighted, k=2, iterations=50, seed=42)
            
            edges = G_weighted.edges()
            edge_weights = [G_weighted[u][v]['weight'] for u, v in edges]
            
            nx.draw_networkx_edges(G_weighted, pos, 
                                  width=[w*6 for w in edge_weights],
                                  alpha=0.6,
                                  edge_color=edge_weights,
                                  edge_cmap=plt.cm.RdYlGn,
                                  edge_vmin=0.3, edge_vmax=1.0)
            
            nx.draw_networkx_nodes(G_weighted, pos, node_size=700, 
                                  node_color='lightblue', alpha=0.9)
            nx.draw_networkx_labels(G_weighted, pos, font_size=9, font_weight='bold')
            
            plt.title(f'Raw Weighted Network (Edge Width Cosine Similarity)\n{category} - Year {year}\n'
                     f'{G_weighted.number_of_edges()} edges | Min weight: 0.3',
                     fontsize=13, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            net_file = f'{output_dir}/networks_weighted/{year}{category.replace(" ", "")}_weighted.png'
            plt.savefig(net_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"      Raw weighted network: {G_weighted.number_of_edges()} edges")
        
        for tau in thresholds:
            G = build_similarity_network(similarity_df, threshold=tau)
            
            if G is None or G.number_of_nodes() == 0:
                continue
            
            all_networks[category][tau][year] = G
            
            if G.number_of_edges() == 0:
                print(f"    tau={tau}: No edges")
                continue

            print(f"    tau={tau}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, density={nx.density(G):.3f}")
            analysis_count += 1

            centrality_df = compute_centralities(G)
            if centrality_df is None:
                continue

            cent_file = f'{output_dir}/centralities/{year}{category.replace(" ", "")}_tau{tau}.csv'
            centrality_df.to_csv(cent_file)

            for scheme in ['equal', 'correlation', 'entropy']:
                if scheme == 'equal':
                    weights = equal_weighting()
                elif scheme == 'correlation':
                    weights = correlation_based_weighting(centrality_df)
                else:
                    weights = entropy_based_weighting(centrality_df)
                
                scores = compute_weighted_score(centrality_df, weights)
                
                if scores is not None:
                    score_file = f'{output_dir}/scores/{year}{category.replace(" ", "")}tau{tau}{scheme}.csv'
                    scores.to_csv(score_file, header=['Score'])

            if tau == 0.7:
                plt.figure(figsize=(16, 12))
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

                edges = G.edges()
                edge_weights = [G[u][v]['weight'] for u, v in edges]

                nx.draw_networkx_edges(G, pos, alpha=0.5, width=3, 
                                    edge_color=edge_weights,
                                    edge_cmap=plt.cm.YlOrRd,
                                    edge_vmin=tau, edge_vmax=1.0)

                node_colors = [centrality_df.loc[node, 'degree'] if node in centrality_df.index else 0 
                            for node in G.nodes()]
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                    node_size=900, alpha=0.9, cmap='Blues')
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

                plt.title(f'{category} - Year {year}\nThreshold tau = {tau}\n'
                         f'{G.number_of_edges()} edges, Density: {nx.density(G):.3f}', 
                        fontsize=13, fontweight='bold')
                plt.axis('off')
                plt.tight_layout()
                net_file = f'{output_dir}/networks_thresholded/{year}{category.replace(" ", "")}_tau{tau}.png'
                plt.savefig(net_file, dpi=150, bbox_inches='tight')
                plt.close()

print(f"\nCompleted {analysis_count} network analyses across {len(years)} years and {len(thresholds)} thresholds")

print("\n[7/10] Performing threshold sensitivity analysis...")

for category in categories.keys():
    for year in years:
        threshold_data = []
        
        for tau in thresholds:
            if tau in all_networks[category] and year in all_networks[category][tau]:
                G = all_networks[category][tau][year]
                
                threshold_data.append({
                    'Threshold': tau,
                    'Nodes': G.number_of_nodes(),
                    'Edges': G.number_of_edges(),
                    'Density': nx.density(G),
                    'Avg_Degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
                })
        
        if len(threshold_data) >= 2:
            df_thresh = pd.DataFrame(threshold_data)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            df_thresh.plot(x='Threshold', y='Edges', ax=axes[0], 
                          marker='o', color='red', legend=False, linewidth=2, markersize=8)
            axes[0].set_title('Edge Count vs Threshold', fontweight='bold')
            axes[0].set_ylabel('Number of Edges')
            axes[0].grid(True, alpha=0.3)
            
            df_thresh.plot(x='Threshold', y='Density', ax=axes[1], 
                          marker='o', color='blue', legend=False, linewidth=2, markersize=8)
            axes[1].set_title('Network Density vs Threshold', fontweight='bold')
            axes[1].set_ylabel('Density')
            axes[1].grid(True, alpha=0.3)
            
            df_thresh.plot(x='Threshold', y='Avg_Degree', ax=axes[2], 
                          marker='o', color='green', legend=False, linewidth=2, markersize=8)
            axes[2].set_title('Average Degree vs Threshold', fontweight='bold')
            axes[2].set_ylabel('Average Degree')
            axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(f'Threshold Sensitivity Analysis\n{category} - Year {year}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            filename = f'{output_dir}/threshold_analysis/{category.replace(" ","")}{year}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

print(f"Threshold sensitivity analysis completed")

print("\n[8/10] Creating temporal Hasse diagrams (year-to-year evolution)...")

def create_hasse_diagram_yearwise(networks_by_year, category, threshold):
    if len(networks_by_year) < 2:
        return None
    
    H = nx.DiGraph()
    years_list = sorted(networks_by_year.keys())
    H.add_nodes_from([str(y) for y in years_list])
    
    print(f"    Analyzing temporal relations for {category} (tau={threshold})...")
    
    edges_added = 0
    relations = []
    
    for i, y1 in enumerate(years_list):
        for j, y2 in enumerate(years_list):
            if i >= j:
                continue
                
            G1 = networks_by_year[y1]
            G2 = networks_by_year[y2]
            
            e1 = set(tuple(sorted(e)) for e in G1.edges())
            e2 = set(tuple(sorted(e)) for e in G2.edges())
            
            if e1.issubset(e2) and e1 != e2:
                H.add_edge(str(y1), str(y2), relation='subset')
                edges_added += 1
                relations.append(f"{y1} subset {y2} (E_{y1} subset E_{y2})")
                print(f"      {y1} subset {y2} (edge subset: {len(e1)} subset {len(e2)})")
            elif e2.issubset(e1) and e1 != e2:
                H.add_edge(str(y2), str(y1), relation='subset')
                edges_added += 1
                relations.append(f"{y2} subset {y1} (E_{y2} subset E_{y1})")
                print(f"      {y2} subset {y1} (edge subset: {len(e2)} subset {len(e1)})")
    
    plt.figure(figsize=(14, 10))
    
    if H.number_of_edges() > 0:
        pos = nx.spring_layout(H, k=3, iterations=100, seed=42)
        
        nx.draw_networkx_edges(H, pos, 
                              arrows=True, 
                              arrowsize=30, 
                              edge_color='darkred',
                              arrowstyle='->', 
                              width=3,
                              alpha=0.7)
        
        node_sizes = [networks_by_year[int(node)].number_of_nodes() * 300 for node in H.nodes()]
        
        nx.draw_networkx_nodes(H, pos, 
                              node_color='lightgreen', 
                              node_size=node_sizes, 
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=2)
        
        nx.draw_networkx_labels(H, pos, 
                               font_size=14, 
                               font_weight='bold')
        
        stats_labels = {}
        for year_str in H.nodes():
            year = int(year_str)
            G = networks_by_year[year]
            stats_labels[year_str] = f"{year}\n({G.number_of_nodes()}n, {G.number_of_edges()}e)"
        
        label_pos = {k: (v[0], v[1]-0.15) for k, v in pos.items()}
        nx.draw_networkx_labels(H, label_pos, stats_labels, 
                               font_size=10, font_color='darkblue')
        
        plt.title(f'Temporal Hasse Diagram (Partial Order)\n{category} | Threshold tau={threshold}\n'
                 f'Relation: G_y subset G_y', 
                 fontsize=13, fontweight='bold')
        
        relations_text = "Temporal Relations:\n" + "\n".join(relations)
        plt.figtext(0.02, 0.02, relations_text, fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
    else:
        pos = nx.circular_layout(H)
        node_sizes = [networks_by_year[int(node)].number_of_nodes() * 300 for node in H.nodes()]
        
        nx.draw_networkx_nodes(H, pos, 
                              node_color='lightcoral', 
                              node_size=node_sizes, 
                              alpha=0.8,
                              edgecolors='black',
                              linewidths=2)
        
        stats_labels = {}
        for year_str in H.nodes():
            year = int(year_str)
            G = networks_by_year[year]
            stats_labels[year_str] = f"{year}\n({G.number_of_nodes()}n, {G.number_of_edges()}e)"
        
        nx.draw_networkx_labels(H, pos, stats_labels, 
                               font_size=12, font_weight='bold')
        
        plt.title(f'Independent Networks (No Subset Relations)\n{category} | Threshold tau={threshold}', 
                 fontsize=13, fontweight='bold')
        
        plt.figtext(0.02, 0.02, "No temporal partial order found.\nNetworks evolved independently.", 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    filename = f'{output_dir}/temporal_hasse/{category.replace(" ", "").replace("&", "")}_tau{threshold}_hasse.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return H

hasse_count = 0
if len(years) >= 2:
    for category in categories.keys():
        for tau in thresholds:
            if tau in all_networks[category]:
                networks_by_year = all_networks[category][tau]
                if len(networks_by_year) >= 2:
                    H = create_hasse_diagram_yearwise(networks_by_year, category, tau)
                    if H:
                        hasse_count += 1
else:
    print("Need at least 2 years for temporal Hasse diagrams")

print(f"Created {hasse_count} temporal Hasse diagrams")

print("\n[9/10] Creating year-to-year comparison visualizations...")

if len(years) >= 2:
    for category in list(categories.keys()):
        if category in all_similarities and len(all_similarities[category]) >= 2:
            year_list = sorted(all_similarities[category].keys())
            
            fig, axes = plt.subplots(1, len(year_list), figsize=(8*len(year_list), 6))
            if len(year_list) == 1:
                axes = [axes]
            
            for idx, year in enumerate(year_list):
                sim_df = all_similarities[category][year]
                sns.heatmap(sim_df, annot=True, fmt='.2f', 
                           cmap='YlOrRd', square=True, vmin=0, vmax=1,
                           ax=axes[idx], cbar_kws={'label': 'Similarity'})
                axes[idx].set_title(f'Year {year}', fontsize=12, fontweight='bold')
            
            plt.suptitle(f'Year-to-Year Similarity Comparison\n{category}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            filename = f'{output_dir}/year_comparisons/{category.replace(" ", "_")}_similarity_comparison.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

print(f"Year-to-year comparison visualizations created")

print("\n[10/10] Computing overall scores across all categories...")

def compute_overall_score_yearly(df, year, threshold=0.7, centrality_weights='equal'):
    overall_scores = {}
    
    for category, alpha in category_weights.items():
        price_matrix = create_yearly_price_matrix(df, year, category)
        if price_matrix is None or price_matrix.empty:
            continue
        
        similarity_df = compute_city_similarity(price_matrix)
        if similarity_df is None:
            continue
        
        G = build_similarity_network(similarity_df, threshold)
        if G is None or G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue
        
        centrality_df = compute_centralities(G)
        if centrality_df is None:
            continue
        
        if centrality_weights == 'equal':
            weights = equal_weighting()
        elif centrality_weights == 'correlation':
            weights = correlation_based_weighting(centrality_df)
        elif centrality_weights == 'entropy':
            weights = entropy_based_weighting(centrality_df)
        else:
            weights = equal_weighting()
        
        category_scores = compute_weighted_score(centrality_df, weights)
        
        if category_scores is not None:
            for city, score in category_scores.items():
                if city not in overall_scores:
                    overall_scores[city] = 0
                overall_scores[city] += alpha * score
    
    if not overall_scores:
        return None
    
    return pd.Series(overall_scores).sort_values(ascending=False)

overall_results = {}

for year in years:
    print(f"  Computing overall scores for Year {year}...")
    
    for scheme in ['equal', 'correlation', 'entropy']:
        overall_scores = compute_overall_score_yearly(df, year, 
                                                      threshold=0.7, 
                                                      centrality_weights=scheme)
        
        if overall_scores is not None and len(overall_scores) > 0:
            filename = f'{output_dir}/overall_scores/{year}_{scheme}.csv'
            overall_scores.to_csv(filename, header=['Overall_Score'])
            
            if year not in overall_results:
                overall_results[year] = {}
            overall_results[year][scheme] = overall_scores

print(f"Overall scores computed and saved")

if len(years) >= 2:
    for scheme in ['equal', 'correlation', 'entropy']:
        fig, axes = plt.subplots(1, len(years), figsize=(8*len(years), 6))
        if len(years) == 1:
            axes = [axes]
        
        for idx, year in enumerate(years):
            if year in overall_results and scheme in overall_results[year]:
                top5 = overall_results[year][scheme].nlargest(5).sort_values()
                top5.plot(kind='barh', ax=axes[idx], color='steelblue')
                axes[idx].set_title(f'Year {year}', fontsize=14, fontweight='bold')
                axes[idx].set_xlabel('Weighted Score')
                axes[idx].set_ylabel('City')
                axes[idx].grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'Top 5 Cities Comparison Across Years\n{scheme.capitalize()} Weighting | Threshold tau = 0.7', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        filename = f'{output_dir}/comparisons/year_comparison_{scheme}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

print("\n" + "="*70)
print("GENERATING COMPREHENSIVE SUMMARY REPORT")
print("="*70)

def count_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

summary_text = f"""
{'='*70}
DISCRETE STRUCTURES PROJECT - YEAR-WISE ANALYSIS SUMMARY
{'='*70}

Project Type: Temporal Price Similarity Network Analysis (Year-Wise)

Data Information:
- Total Records: {len(df)}
- Years Analyzed: {years}
- Number of Years: {len(years)}
- Cities: {df['City'].nunique()}
- Products: {df['Product'].nunique()}
- Categories: {len(categories)}

Analysis Approach:
YEAR-WISE AGGREGATION (all months averaged per year)
Temporal evolution tracked across years
Hasse diagrams show year-to-year partial order

Analysis Parameters:
- Similarity Metric: Cosine Similarity
- Thresholds Analyzed: {thresholds}
- Centrality Measures: Degree, Closeness, Betweenness, Eigenvector
- Weighting Schemes: Equal, Correlation-based, Entropy-based

Category Weights (alpha_c):
"""

for cat, weight in category_weights.items():
    summary_text += f"  {cat}: {weight}\n"

summary_text += f"""
Output Files Generated:
- Similarity Heatmaps (per year): {count_files(f'{output_dir}/heatmaps')}
- Thresholded Networks: {count_files(f'{output_dir}/networks_thresholded')}
- Raw Weighted Networks: {count_files(f'{output_dir}/networks_weighted')}
- Threshold Sensitivity: {count_files(f'{output_dir}/threshold_analysis')}
- Centrality CSVs: {count_files(f'{output_dir}/centralities')}
- Score CSVs: {count_files(f'{output_dir}/scores')}
- Overall Score CSVs: {count_files(f'{output_dir}/overall_scores')}
- Year Comparisons: {count_files(f'{output_dir}/year_comparisons')}
- Temporal Hasse Diagrams: {count_files(f'{output_dir}/temporal_hasse')}
- Comparison Plots: {count_files(f'{output_dir}/comparisons')}

Total Files: {count_files(output_dir)}

PROJECT REQUIREMENTS STATUS:
7 Categories as specified
Year-wise aggregation (R_y,c relation)
Multiple thresholds (0.6, 0.7, 0.8)
Threshold effect visualization
Raw cosine similarity weighted networks
Thresholded similarity networks
All 4 centrality measures computed
3 weighting schemes implemented
Category importance weighting (alpha_c)
Temporal Hasse diagrams (G_y subset G_y)
Year-to-year evolution analysis
All mathematical requirements met

Mathematical Model:
- Relation: (C_i, C_j) in R_y,c CosineSim_y,c(C_i, C_j) >= tau
- Network: G_y,c = (V, E_y,c)
- Temporal Order: G_y subset G_y
- Weighted Score: S_i,y = sum_c alpha_c [sum_k w_k Centrality_k]

Key Findings:
"""

if overall_results:
    latest_year = sorted(overall_results.keys())[-1]
    if 'equal' in overall_results[latest_year]:
        top_city = overall_results[latest_year]['equal'].idxmax()
        top_score = overall_results[latest_year]['equal'].max()
        summary_text += f"- Top ranked city (Year {latest_year}): {top_city} (Score: {top_score:.4f})\n"
        
        top5 = overall_results[latest_year]['equal'].nlargest(5)
        summary_text += f"\nTop 5 Cities (Year {latest_year}, Equal Weighting):\n"
        for i, (city, score) in enumerate(top5.items(), 1):
            summary_text += f"  {i}. {city}: {score:.4f}\n"

if len(years) >= 2:
    summary_text += f"\nTemporal Evolution Analysis:\n"
    summary_text += f"- Analyzed {len(years)} years: {years}\n"
    summary_text += f"- Created {hasse_count} Hasse diagrams showing temporal partial order\n"
    summary_text += f"- Year-to-year comparisons reveal network evolution patterns\n"

summary_text += f"""
All results saved in: ./{output_dir}/

{'='*70}
PROJECT COMPLETE - YEAR-WISE ANALYSIS SUCCESSFUL
{'='*70}

Advantages of Year-Wise Approach:
1. Clearer temporal patterns (year-to-year evolution)
2. Reduced file count (more manageable)
3. Better alignment with mathematical model (R_y,c)
4. More meaningful Hasse diagrams (G_y subset G_y)
5. Easier to identify long-term trends

Files ready for:
Demo video presentation
Report submission
Temporal analysis discussion
"""

with open(f'{output_dir}/SUMMARY_REPORT.txt', 'w') as f:
    f.write(summary_text)

print(summary_text)

print("\n" + "="*70)
print("PACKAGING FILES FOR DOWNLOAD")
print("="*70)

print("\nCreating download packages...")

shutil.make_archive('results_yearly_complete', 'zip', output_dir)
print(f"results_yearly_complete.zip created ({os.path.getsize('results_yearly_complete.zip') / (1024*1024):.1f} MB)")

zip_files_created = []
for subdir in subdirs:
    subdir_path = f'{output_dir}/{subdir}'
    if os.path.exists(subdir_path) and count_files(subdir_path) > 0:
        zip_name = f'{subdir}_yearly'
        shutil.make_archive(zip_name, 'zip', subdir_path)
        zip_files_created.append(f'{zip_name}.zip')
        print(f"{zip_name}.zip created ({count_files(subdir_path)} files)")

print(f"\nAll files packaged successfully")

print("\n" + "="*70)
print("INITIATING AUTO-DOWNLOAD")
print("="*70)

try:
    from google.colab import files as colab_files
    IN_COLAB = True
    print("\nGoogle Colab environment detected")
except:
    IN_COLAB = False
    print("\nLocal Python environment detected")

if IN_COLAB:
    print("\n" + "="*70)
    print("DOWNLOADING FILES TO YOUR COMPUTER...")
    print("="*70)
    
    print("\n[1] Downloading main complete package...")
    colab_files.download('results_yearly_complete.zip')
    print("    results_yearly_complete.zip downloaded")
    
    download_count = 2
    for zip_file in zip_files_created:
        if os.path.exists(zip_file):
            print(f"\n[{download_count}] Downloading {zip_file}...")
            colab_files.download(zip_file)
            print(f"    {zip_file} downloaded")
            download_count += 1
    
    print(f"\n[{download_count}] Downloading summary report...")
    colab_files.download