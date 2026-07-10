"""
Script to curate the 10-task Team-SWE benchmark sequence by building custom DAGs 
from chronological diff-overlaps in dense repositories.
"""
import logging
import pandas as pd
import networkx as nx
from typing import List, Dict, Set
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DAG_Curation")

def extract_touched_files(patch: str) -> Set[str]:
    """Extracts the set of files touched (added, modified, deleted) in a diff patch."""
    files = set()
    if not isinstance(patch, str): return files
    for line in patch.split('\n'):
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            files.add(line[6:].strip())
    return files

def extract_added_lines(patch: str) -> Set[str]:
    added = set()
    if not isinstance(patch, str): return added
    for line in patch.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            clean_line = line[1:].strip()
            if clean_line and len(clean_line) > 3:
                added.add(clean_line)
    return added

def extract_deleted_lines(patch: str) -> Set[str]:
    deleted = set()
    if not isinstance(patch, str): return deleted
    for line in patch.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            clean_line = line[1:].strip()
            if clean_line and len(clean_line) > 3:
                deleted.add(clean_line)
    return deleted

def build_repo_dag(repo_df: pd.DataFrame) -> nx.DiGraph:
    """
    Builds a DAG for a single repository based on chronological file overlaps.
    repo_df must be sorted chronologically by created_at.
    """
    G = nx.DiGraph()
    
    # Extract file sets for all tasks upfront to save time
    task_files = {}
    for idx, row in repo_df.iterrows():
        instance_id = row['instance_id']
        files = extract_touched_files(row.get('patch', ''))
        task_files[instance_id] = files
        G.add_node(instance_id) # ensure node exists even if isolated

    # We only want directed edges i -> j where i < j chronologically
    tasks = repo_df['instance_id'].tolist()
    
    edges_added = 0
    for j in range(1, len(tasks)):
        task_j = tasks[j]
        files_j = task_files[task_j]
        if not files_j: continue
        
        for i in range(j):
            task_i = tasks[i]
            files_i = task_files[task_i]
            
            # If there is any file overlap, we draw a dependency edge
            if files_i.intersection(files_j):
                G.add_edge(task_i, task_j)
                edges_added += 1
                
    return G

def score_inversions(chain: List[str], patch_lookup: Dict[str, str]) -> int:
    """Scores a chain based on temporal line-level inversions."""
    inversions = 0
    for i in range(len(chain)):
        task_i = chain[i]
        added_in_i = extract_added_lines(patch_lookup.get(task_i, ""))
        if not added_in_i: continue
            
        for j in range(i + 1, len(chain)):
            task_j = chain[j]
            deleted_in_j = extract_deleted_lines(patch_lookup.get(task_j, ""))
            
            overlap = added_in_i.intersection(deleted_in_j)
            if overlap:
                inversions += len(overlap)
                
    return inversions

def main():
    logger.info("Starting Custom DAG Curation...")
    
    # 1. Load Data
    url_base = "https://huggingface.co/datasets/jiayuanz3/SWEContextBench/resolve/main/data/SWEContextBench_Related.parquet"
    try:
        base_df = pd.read_parquet(url_base)
        logger.info(f"Loaded {len(base_df)} base tasks.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # To sort chronologically, ensure created_at is datetime
    base_df['created_at'] = pd.to_datetime(base_df['created_at'], errors='coerce')
    
    # 2. Iterate Dense Repos
    repo_counts = base_df['repo'].value_counts()
    dense_repos = repo_counts[repo_counts >= 10].index.tolist()
    
    logger.info(f"Found {len(dense_repos)} repos with >= 10 tasks.")
    
    all_long_chains = []
    patch_lookup = dict(zip(base_df['instance_id'], base_df['patch']))
    
    print("\n" + "="*50)
    print("DAG LENGTH ANALYSIS PER REPOSITORY")
    print("="*50)
    
    for repo in dense_repos:
        repo_df = base_df[base_df['repo'] == repo].copy()
        # Sort chronologically (oldest first)
        repo_df = repo_df.sort_values(by='created_at')
        
        # Build DAG
        G = build_repo_dag(repo_df)
        
        # Analyze DAG
        components = list(nx.weakly_connected_components(G))
        max_len = 0
        repo_long_chains = []
        
        for comp in components:
            subgraph = G.subgraph(comp)
            if not nx.is_directed_acyclic_graph(subgraph):
                continue
            
            try:
                longest_path = nx.dag_longest_path(subgraph)
                l = len(longest_path)
                max_len = max(max_len, l)
                if l >= 10:
                    repo_long_chains.append(longest_path)
                    all_long_chains.append(longest_path)
            except nx.NetworkXUnfeasible:
                pass
                
        print(f"Repo: {repo:<25} | Tasks: {len(repo_df):<4} | DAG Max Chain Length: {max_len}")
        if repo_long_chains:
            logger.debug(f"-> {repo} has {len(repo_long_chains)} chains >= 10 tasks.")
            
    # 3. Score Long Chains for Inversions
    print("\n" + "="*50)
    print("INVERSION SCORING (Chains >= 10)")
    print("="*50)
    
    if not all_long_chains:
        logger.warning("No chains of length >= 10 were found across any repository.")
        return
        
    logger.info(f"Found {len(all_long_chains)} total chains across all repos of length >= 10. Scoring...")
    
    best_chain = None
    max_score = -1
    
    for idx, chain in enumerate(all_long_chains):
        # We only need exactly a 10-task window to evaluate.
        # We slide a 10-task window over the chain and find the window with max inversions.
        for w_start in range(len(chain) - 9):
            window = chain[w_start:w_start+10]
            score = score_inversions(window, patch_lookup)
            
            if score > max_score:
                max_score = score
                best_chain = window
                
    if best_chain:
        logger.info(f"Selected best 10-task sequence with inversion score {max_score}.")
        print(f"WINNER: {best_chain}")
        output_file = "team_swe_cluster.json"
        pd.Series(best_chain).to_json(output_file, orient="values")
        logger.info(f"Saved to {output_file}")
    else:
        logger.warning("No valid inversion sequences found.")

if __name__ == "__main__":
    main()
